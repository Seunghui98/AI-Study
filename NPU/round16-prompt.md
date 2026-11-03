# 16회차 측정 요청 (세 번째): int8 KV 하나의 chunk 경로 + 느린 첫 토큰

두 번째 시도(`2169d3134`)도 0번에서 멈춤: 세 번째 chunk에서 "layer0_attention could not take its rows [0, 1024) back from the
decode cache". 층의 첫 호출(scale 보정 경로)은 decode cache로 seed했지만, 두 번째 chunk부터 쓰는 f32 attention 경로는 seed 없이
돌아가서 decode cache가 512행에 머물렀음. `6d89d157f`에서 f32 경로도 seed함. 그리고 int8only에서 seed 없이 끝난 호출이 있으면
한 chunk 뒤가 아니라 그 자리에서 행 범위를 적고 멈춤. 이 빈틈은 default14의 chunk prompt에도 있었음(마지막 chunk가 prefill 중에
seed하지 않고 첫 토큰으로 미룸) → 이번부터 c-d14도 마지막 chunk에서 seed함.

첫 시도 기록:

첫 시도(`cfb6c08a5`)는 0번에서 멈춤: 두 번째 chunk(위치 512)를 `mha_core`의 안전장치("a multi-row forwarding at 512 after the
HTP decode hook took over the KV cache")가 막음. int8only가 prefill 호출마다 "DSP가 넘겨받음" 표시를 켰기 때문이고, 이 표시는
지난 CPU 행이 낡았다는 뜻이라 다음 chunk를 거부한다. int8only는 앞 행을 decode cache에서 다시 채우므로 막을 이유가 없음.
`2169d3134`에서 prefill 호출은 이 표시를 켜지 않음 (decode가 시작된 뒤의 두 번째 prompt는 전처럼 거부).
기본 설정에서도 INIT_SEQ_LEN보다 긴 prompt는 chunk가 나뉘므로, 이 문제는 긴 prompt 전부에 해당했음.

보는 것:
1. **chunk 경로**: 두 번째 chunk부터 앞 행을 decode cache에서 다시 채우는 경로(fill, reseed, scale이 커질 때 old/new 비율로
   재양자화). `NNTR_PREFILL_CHUNK=512`로 강제 (p1k 3개, p2k 4개 chunk).
2. **끝이 1행·정렬 안 된 chunk**: `NNTR_PREFILL_CHUNK=1089`로 p1k(1090토큰)를 1089 + 1행으로 나눔. fill이 32행 tile 중간에서
   끝나고 마지막 chunk가 1행인 경우.
3. **느린 첫 토큰**: 15회차 int8only 10회 중 3회가 174.6~185.1 ms (packet 101~103 ms). default14는 10회 모두 153~159 ms.

## ① 측정 시킬 때 (그대로 붙여넣기)

```text
nntrainer HTP(Gemma4-26B-A4B, S26U) 16회차 (세 번째). 코드 수정 없이 측정·보고만.
빌드: fork Seunghui98/nntrainer, 브랜치 htp/first-token-kv-int8. git fetch 후 git rev-parse --short HEAD 가 6d89d157f 인지 확인
(다르면 측정하지 말고 해시만 보고). 2169d3134 뒤로 바뀐 것은 Applications/CausalLM/layers/htp_mha_core.cpp 하나 (IDL 그대로).
평소처럼 빌드한 뒤 libcausallm.so를 새로 올릴 것. skel은 지난 시도에 올린 930b6d21…, libnntrainer.so는 344f3c3b…를
그대로 쓰고 기기 md5만 확인 (빌드로 skel md5가 또 바뀌어도 소스가 같으므로 올리지 말고 새 md5만 적을 것).
새 libcausallm.so md5를 적을 것.
조건: 15회차와 같음 (R5KL309ALRK, gemma4-26b-a4b-tl2-rotated, C=16, greedy, EOS 끔, 128토큰, max_seq_len 2560,
NNTR_HTP_PAGE_CACHE=0, 매 실행 전 35 ℃ 이하, run.sh).

셀:
  c-int8    NNTR_PREFILL_CHUNK=512
  c-d14     NNTR_PREFILL_CHUNK=512 NNTR_HTP_KV_INT8_ONLY=0
  u-int8    NNTR_PREFILL_CHUNK=1089
  u-d14     NNTR_PREFILL_CHUNK=1089 NNTR_HTP_KV_INT8_ONLY=0
  int8only  (스위치 없음)
  default14 NNTR_HTP_KV_INT8_ONLY=0

## 0. 먼저 확인 (1회)
p2k를 c-int8로 1회 (NNTR_HTP_PROFILE=1). pool_sync 줄 원문, prefill_seeds, prefill_fills, "attn_m1: registered" 두 줄 원문,
생성 텍스트 앞부분.
- 기대: prefill_seeds=120, prefill_fills=90.
- "without seeding its decode cache" 가 나오면 그 줄 원문과 직전 30줄, 명령만 보고.
- prefill_fills가 0이면 chunk가 안 나뉜 것 → A~D 하지 말고 그 로그의 앞 50줄과 명령만 보고.
- "FATAL ERROR" / "fell off the int8 KV path" / "could not seed" / "could not take its rows" / "not seeded" / "not filled" /
  FastRPC·DSP 에러 / crash / hang이 나오면 A~D 하지 말고 그 메시지 원문과 직전 30줄, 명령만 보고.
문제없으면 A로 진행.

## A. chunk 경로 정확도 (NNTR_HTP_PROFILE 끔, NNTR_HTP_ATTN_TRACE=1: scale 증가 줄을 찍기 위함)
13회차 acc/ 의 cpu.ids 그대로. 13개 prompt (a01~a11, p1k, p2k)마다 c-int8, c-d14 각 1회 (같은 prompt의 두 셀은 연달아, 순서 번갈아):
  NNTR_HTP_ATTN_TRACE=1 NNTR_PPL_DECODE=$Q/<prompt>.cpu.ids → r16/logs/<prompt>__c-int8.log, <prompt>__c-d14.log
13회차 <prompt>__ref.log 를 r16/logs로 복사한 뒤 PC에서:
  python3 -I <AI-Study>/NPU/ppl_eval.py r16/logs c-d14 c-int8
출력 원문 전체. prompt마다: prefill 토큰 수, 두 셀의 "[PPL] decode tokens=" 줄 원문, 두 셀의 "[PPL] decode step=" 줄이
완전히 같은지 (같음 / 다른 줄 수와 처음 다른 step), "q2 scales grew" 줄 수와 처음 몇 줄 원문 (없으면 0).
c-int8의 step 줄이 15회차 int8only(= 13회차 default)와 같은지도 (512토큰 이하 prompt는 한 블록이라 같아야 함).

## B. chunk 경로 속도·메모리 (NNTR_HTP_PROFILE=1, NNTR_PPL_DECODE 없이)
p1k · p2k, c-int8 · c-d14 각 3회, 셀 순서 번갈아.
실행마다: prefill TPS, generation TPS, first_token ms, timeline 줄, pool_sync 줄의 prefill_seeds / prefill_fills /
prefill_fill_ms, peak MiB, peak by phase (load / prefill / decode), ids md5 (c-int8과 c-d14가 같은지, 15회차 int8only와 같은지).
중앙값 표, 같은 rep 차이표 (c-int8 − c-d14 의 prefill TPS · prefill_fill_ms · 단계별 peak).

## C. 느린 첫 토큰 (NNTR_HTP_PROFILE=1)
p1k · p2k, int8only · default14 각 8회, 셀 순서 번갈아 (int8only → default14 → default14 → int8only → ...).
실행마다: first_token ms, timeline 줄 (graph_init / to_token / pool_sync / to_packet / packet), token1 us / misses,
warm registered / cancelled, mem_available_mib (pool_sync 줄), peak MiB.
first_token이 170 ms를 넘은 실행은 그 실행의 timeline 줄, pool_sync 줄, token1 줄, warm 줄 원문.
중앙값 표와 170 ms 넘은 횟수 표 (셀 × prompt). 그리고 p1k · p2k 각 셀 첫 실행의 "attn_m1: registered" 두 줄 원문.

## D. 끝이 1행인 chunk (NNTR_HTP_PROFILE=1, NNTR_HTP_ATTN_TRACE=1)
p1k만, u-int8 · u-d14 각 1회, NNTR_PPL_DECODE=$Q/p1k.cpu.ids → r16/logs/p1k__u-int8.log, p1k__u-d14.log.
두 셀의 "[PPL] decode tokens=" 줄 원문, step 줄이 같은지, u-int8의 pool_sync 줄 원문 (기대: prefill_seeds=60, prefill_fills=30),
"q2 scales grew" 줄 수. 에러가 나면 메시지 원문과 직전 30줄.

## 규칙
- 실제로 돌린 것만 "확인됨". 못 돌린 항목은 못 돌렸다고 적을 것.
- 숫자를 추정하거나 지난 결과로 채우지 말 것.
- 실패하면 고치려 하지 말고 에러와 명령을 그대로 보고할 것. A~D 중 하나가 에러로 멈추면 그 항목만 멈추고
  나머지는 진행하되, 에러 원문과 직전 30줄을 적을 것.
- 요청서와 다르게 한 것이 있으면 "이상한 점"에 이유와 함께 적을 것.
```

## ② 결과 정리시킬 때

```text
16회차 결과를 0, A, B, C, D 형식 그대로 정리해줘. 맨 앞에 환경(빌드 해시, 새로 올린 파일과 md5, 기기, 모델, 온도, 날짜),
0은 확인 결과와 원문 줄, A는 prompt별 표(prompt | prefill 토큰 | chunk 수 | c-d14 nll_sum / top1 | c-int8 nll_sum / top1 |
step 줄 같음? | scales grew 줄 수) + ppl_eval.py 출력 원문, B는 실행별 표 + 중앙값 표 + 같은 rep 차이표, C는 실행별 표 +
중앙값 표 + 170 ms 넘은 횟수 표 + 넘은 실행의 원문 줄 + registered 줄, D는 두 셀 결과와 원문 줄,
맨 뒤에 "이상한 점"과 원본 로그 경로.
```

## 결과 해석

- **A, step 줄이 c-d14와 같음**: chunk 사이 scale이 커지지 않았으면 두 셀은 같은 int8 바이트를 써야 함 (c-d14는 fp16에서
  같은 scale로 다시 양자화, c-int8은 int8을 그대로 채움). 다르면 scale이 커진 prompt인지("q2 scales grew") 먼저 볼 것:
  커진 경우에는 c-int8이 int8을 한 번 더 반올림하므로 작게 다를 수 있음 → ppl_eval verdict로 판정.
- **A, scale이 안 커졌는데 다름**: fill 경로 버그 → 처음 다른 step과 prompt로 좁힘.
- **B**: c-int8은 prefill_fills = 30 × (chunk 수 − 1) (p1k 60, p2k 90), prefill_seeds = 30 × chunk 수. c-d14는 fill 0,
  prefill_seeds 30 (마지막 chunk만). prefill_fill_ms가 prefill 시간에서
  차지하는 몫과 prefill TPS 차이. peak는 c-int8이 c-d14보다 낮아야 함.
- **C**: 170 ms 넘는 실행이 int8only에만 몰리면 원인을 찾음 (token1 misses, warm cancelled, mem_available 비교).
  두 셀에 고르게 나오면 기기 변동. registered 줄: int8only의 sliding 층 max_seq가 prompt + 생성분(p1k 1280, p2k 2176)인지,
  default14가 2560인지 (15회차 decode peak 감소의 원인 확인).
- **D**: 에러 없이 끝나고 step 줄이 u-d14와 같거나(scale 안 커짐) verdict 수준으로 가까우면 1행·비정렬 경계도 통과.
