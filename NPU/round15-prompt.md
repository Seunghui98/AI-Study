# 15회차 측정 요청: KV를 int8 하나로 (host fp16 KV 없음)

목적: CPU의 fp16 KV 캐시(층마다 max_seq 행, 30층 합 450–550 MiB)를 없애고, 층의 KV를 DSP의 int8 decode cache 하나에만 둔다.
CPU에는 한 step의 행만 잠깐 담는 버퍼 하나(모든 층 공유, 수 MiB)만 남는다. prefill의 int8 cache는 호출마다 decode cache에서
앞 행을 다시 채우고(fill), 그 호출의 행을 붙이고, decode cache로 다시 넘긴다(seed).

기대:
- **출력**: prompt가 한 블록이면(chunk 1개) 13·14회차 default와 비트 단위로 같음 (같은 scale, 같은 int8 바이트). chunk가 여럿이면
  두 번째 chunk부터는 앞 행을 fp16 대신 int8에서 다시 채우므로 작게 다를 수 있음.
- **메모리**: load·prefill 단계 peak 감소 (host fp16 KV가 없음). decode 단계 peak는 p1k에서는 거의 같음. p2k처럼 chunk가 여럿인
  prompt는 sliding 층의 decode cache가 창(1024행) 대신 prompt 전체 행을 들고 있어 **decode 단계가 최대 ~100 MiB 늘 수 있음**. 이번에
  그 크기를 재서 다음 단계(생성 시작 때 decode cache를 창 크기로 줄이기)를 할지 정함.
- **속도**: prefill은 같거나 약간 느림 (두 번째 chunk부터 fill 비용). 첫 토큰은 같음.

## ① 측정 시킬 때 (그대로 붙여넣기)

```text
nntrainer HTP(Gemma4-26B-A4B, S26U) 15회차: int8 KV 하나로 (host fp16 KV 없음). 코드 수정 없이 측정·보고만.

빌드: fork Seunghui98/nntrainer, 브랜치 htp/first-token-kv-int8. git fetch 후 git rev-parse --short HEAD 가 cfb6c08a5 인지 확인
(다르면 측정하지 말고 해시만 보고). IDL이 바뀜 (kv_fill_q 메서드 추가) → 13회차처럼 IDL에서 stub/skel을 다시 생성하는 전체
빌드로 skel, libnntrainer.so, libcausallm.so 모두 새로 올릴 것. stage.sh가 libc++_shared 단계에서 멈추면 md5만 확인.
조건: R5KL309ALRK, models/gemma4-26b-a4b-tl2-rotated, C=16, greedy, EOS 끔, 128토큰, max_seq_len 2560, NNTR_HTP_PAGE_CACHE=0,
매 실행 전 35 ℃ 이하, run.sh.

셀:
  int8only  (스위치 없음 = 새 기본값)
  default14 NNTR_HTP_KV_INT8_ONLY=0              (14회차 default와 같은 동작)
  off       NNTR_HTP_KV_ADOPT=0 NNTR_MOE_WARM=0  (adopt 없음; 자동으로 int8only도 꺼짐)

## A. 동작·정확도 (NNTR_HTP_PROFILE 끔)
13회차 Q 디렉터리의 cpu.ids를 그대로 사용 (ref는 다시 돌리지 말 것). 13개 prompt (a01~a11, p1k, p2k)마다 int8only 1회:
  NNTR_PPL_DECODE=$Q/<prompt>.cpu.ids → r15/logs/<prompt>__int8only.log
그리고 13회차 로그 디렉터리의 <prompt>__ref.log, <prompt>__default.log를 r15/logs로 복사한 뒤 PC에서:
  python3 -I <AI-Study>/NPU/ppl_eval.py r15/logs default int8only
출력 원문 전체. prompt마다: prefill 토큰 수, "[PPL] decode tokens=" 줄 원문 (13회차 default 것과 같이), 13회차 default 로그와
"[PPL] decode step=" 줄들이 완전히 같은지 (diff 결과: 같음 / 다른 줄 수와 처음 다른 step).

## B. 속도·메모리 (NNTR_HTP_PROFILE=1, NNTR_PPL_DECODE 없이)
p1k · p2k, 셀마다 5회, 셀 순서 돌아가며 (int8only → default14 → off → ...).
실행마다: prefill TPS, generation TPS, first_token ms, timeline 줄 (graph_init / to_token / pool_sync / to_packet / packet),
pool_sync 줄의 prefill_fills / prefill_fill_ms 값, prefill_seeds 값, peak MiB, peak by phase (load / prefill / decode 각각),
ids md5가 14회차 같은 셀(int8only는 14회차 default)과 같은지.
중앙값 표, 같은 rep 차이표 (int8only − default14 의 prefill TPS · first_token · 단계별 peak), 14회차와 비교표:
  14회차 default / off: first_token 154.7 / 391.6 · 159.2 / 426.7 ms
rep 1마다 원문: timeline 줄, pool_sync 줄, "peak memory:" / "(honest ..." / "(peak by phase KB: ..." 줄.
int8only에서 확인: "fell off the int8 KV path" / "could not seed" / "could not take its rows" / "not seeded" 없음,
prefill_seeds = 30 × (chunk 수), 텍스트 정상, FastRPC / DSP 에러·crash·hang (init·종료 7줄 외).
가능하면 p2k 첫 실행에서 chunk 수(prefill 블록 수)를 로그에서 확인.

## 규칙
- 실제로 돌린 것만 "확인됨". 못 돌린 항목은 못 돌렸다고 적을 것.
- 숫자를 추정하거나 지난 결과로 채우지 말 것.
- 실패하면 고치려 하지 말고 에러와 명령을 그대로 보고할 것 (int8only가 throw로 멈추면 그 메시지 원문과 직전 30줄).
- 요청서와 다르게 한 것이 있으면 "이상한 점"에 이유와 함께 적을 것.
```

## ② 결과 정리시킬 때

```text
15회차 결과를 A, B 형식 그대로 정리해줘. 맨 앞에 0. 환경(빌드 해시, 새로 올린 파일과 md5, 기기, 모델, 온도, 날짜),
A는 prompt별 표(prompt | prefill 토큰 | default nll_sum / top1 | int8only nll_sum / top1 | step 줄 같음?) + ppl_eval.py 출력 원문,
B는 실행별 표 + 중앙값 표 + 같은 rep 차이표 + 14회차 비교표, 맨 뒤에 "이상한 점"과 원본 로그 경로.
```

## 결과 해석

- **A, 한 블록 prompt의 step 줄이 13회차 default와 완전히 같음**: int8 하나로도 계산이 그대로 (예상).
  다르면: 앞 행을 채우는 경로(fill)가 아닌 다른 곳이 바뀐 것 → 처음 다른 step으로 좁힘.
- **A, chunk 여러 개 prompt (p2k 등)**: verdict가 "not worse"면 int8에서 다시 채워도 무해.
- **B, prefill_fills**: 한 블록 prompt는 0, chunk가 n개면 30 × (n − 1). prefill_fill_ms가 prefill 시간에서 차지하는 몫.
- **B, peak by phase**: load·prefill이 default14보다 줄면 성공. decode 단계가 p2k에서 늘면 그 크기로 다음 단계(생성 시작 때
  sliding 층 decode cache를 창 크기로 줄이기)를 정함.
