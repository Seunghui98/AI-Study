# 16회차 측정 요청: int8 KV 하나의 chunk 경로 + 느린 첫 토큰

15회차에서 확인 못 한 두 가지를 본다. 코드는 15회차와 같음 (`cfb6c08a5`, 다시 빌드할 필요 없음).

1. **chunk 경로**: 15회차는 모든 prompt가 한 블록이라 `prefill_fills=0`. 두 번째 chunk부터 앞 행을 decode cache에서 다시
   채우는 경로(fill, reseed, scale이 커질 때 old/new 비율로 재양자화)가 기기에서 한 번도 돌지 않았다.
   `NNTR_PREFILL_CHUNK=512`로 prefill을 512행씩 나눠 이 경로를 강제로 탄다 (p1k 3개, p2k 4개 chunk).
2. **느린 첫 토큰**: int8only 10회 중 3회가 174.6~185.1 ms (packet 101~103 ms). default14는 10회 모두 153~159 ms.
   5회로는 우연인지 모름 → 횟수를 늘리고, packet 안의 expert miss와 warm 줄을 같이 본다.

## ① 측정 시킬 때 (그대로 붙여넣기)

```text
nntrainer HTP(Gemma4-26B-A4B, S26U) 16회차. 코드 수정 없이 측정·보고만.
빌드: 15회차 그대로 (fork Seunghui98/nntrainer, htp/first-token-kv-int8, cfb6c08a5). 기기의 skel, libnntrainer.so,
libcausallm.so md5가 15회차와 같은지만 확인 (39d3aca0… / 344f3c3b… / aae5c48a…). 다르면 측정하지 말고 보고.
조건: 15회차와 같음 (R5KL309ALRK, gemma4-26b-a4b-tl2-rotated, C=16, greedy, EOS 끔, 128토큰, max_seq_len 2560,
NNTR_HTP_PAGE_CACHE=0, 매 실행 전 35 ℃ 이하, run.sh).

셀 (C = chunk 512):
  c-int8   NNTR_PREFILL_CHUNK=512
  c-d14    NNTR_PREFILL_CHUNK=512 NNTR_HTP_KV_INT8_ONLY=0
  int8only (스위치 없음)
  default14 NNTR_HTP_KV_INT8_ONLY=0

## A. chunk 경로 정확도 (NNTR_HTP_PROFILE 끔, NNTR_HTP_ATTN_TRACE=1: scale 증가 줄을 찍기 위함)
13회차 acc/ 의 cpu.ids 그대로. 13개 prompt (a01~a11, p1k, p2k)마다 c-int8, c-d14 각 1회 (같은 prompt의 두 셀은 연달아, 순서 번갈아):
  NNTR_HTP_ATTN_TRACE=1 NNTR_PPL_DECODE=$Q/<prompt>.cpu.ids → r16/logs/<prompt>__c-int8.log, <prompt>__c-d14.log
13회차 <prompt>__ref.log 를 r16/logs로 복사한 뒤 PC에서:
  python3 -I <AI-Study>/NPU/ppl_eval.py r16/logs c-d14 c-int8
출력 원문 전체. prompt마다: prefill 토큰 수, 두 셀의 "[PPL] decode tokens=" 줄 원문, prefill_seeds 와 prefill_fills 값
(로그에 PROFILE이 꺼져 pool_sync 줄이 없으면 "없음"), 두 셀의 "[PPL] decode step=" 줄이 완전히 같은지
(같음 / 다른 줄 수와 처음 다른 step). "q2 scales grew" 줄 수와 처음 몇 줄 원문 (없으면 0).

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
중앙값 표와 170 ms 넘은 횟수 표 (셀 × prompt).

## 규칙
- 실제로 돌린 것만 "확인됨". 못 돌린 항목은 못 돌렸다고 적을 것.
- 숫자를 추정하거나 지난 결과로 채우지 말 것.
- 실패하면 고치려 하지 말고 에러와 명령을 그대로 보고할 것 (c-int8이 throw로 멈추면 그 메시지 원문과 직전 30줄).
- 요청서와 다르게 한 것이 있으면 "이상한 점"에 이유와 함께 적을 것.
```

## ② 결과 정리시킬 때

```text
16회차 결과를 A, B, C 형식 그대로 정리해줘. 맨 앞에 0. 환경(빌드 해시, 기기 파일 md5, 기기, 모델, 온도, 날짜),
A는 prompt별 표(prompt | prefill 토큰 | c-d14 nll_sum / top1 | c-int8 nll_sum / top1 | step 줄 같음? | scales grew 줄 수)
+ ppl_eval.py 출력 원문, B는 실행별 표 + 중앙값 표 + 같은 rep 차이표, C는 실행별 표 + 중앙값 표 + 170 ms 넘은 횟수 표
+ 넘은 실행의 원문 줄, 맨 뒤에 "이상한 점"과 원본 로그 경로.
```

## 결과 해석

- **A, step 줄이 c-d14와 같음**: chunk 사이 scale이 커지지 않았으면 두 셀은 같은 int8 바이트를 써야 함 (c-d14는 fp16에서
  같은 scale로 다시 양자화, c-int8은 int8을 그대로 채움). 다르면 scale이 커진 prompt인지("q2 scales grew") 먼저 볼 것:
  커진 경우에는 c-int8이 int8을 한 번 더 반올림하므로 작게 다를 수 있음 → ppl_eval verdict로 판정.
- **A, scale이 안 커졌는데 다름**: fill 경로 버그 → 처음 다른 step과 prompt로 좁힘.
- **B**: prefill_fills = 30 × (chunk 수 − 1) (p1k 60, p2k 90), prefill_seeds = 30 × chunk 수. prefill_fill_ms가 prefill 시간에서
  차지하는 몫과 prefill TPS 차이. peak는 c-int8이 c-d14보다 낮아야 함.
- **C**: 170 ms 넘는 실행이 int8only에만 몰리면 원인을 찾음 (token1 misses, warm cancelled, mem_available 비교).
  두 셀에 고르게 나오면 기기 변동.
