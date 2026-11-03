# 17회차 측정 요청: fill HVX화 (긴 prompt prefill 속도)

16회차: chunk 경로는 동작하지만 fill(앞 행을 decode cache에서 prefill int8 cache로 다시 채우기)이 층당 61~80 ms라 c-int8 prefill이
153 / 157 TPS (c-d14 231 / 298). fill이 스칼라 C였음 (원소마다 float 비율·rintf, K를 64 B 간격으로 읽음).

`d283f77b6`에서 fill을 HVX로 바꿈:
- K: decode 형식(64위치 전치 tile)에서 seed의 byte zip 6단을 vdeal 6단으로 되돌려 위치마다 한 vector → masters
- V: 행 그대로 vector 복사, colsum: vrmpy
- scale이 커졌을 때의 비율: Q15 정수 `(q·r + 2^14) >> 15` (vector). 스칼라 기준도 같은 식으로 바꿈 → 16회차 c-int8과
  scale이 커진 prompt에서 아주 작게 다를 수 있음 (rintf와 반올림 경계에서만)
- host(hvx_emu): HVX fill = 스칼라 fill 바이트 동일 (head_dim 256 / 512, tile 중간에서 시작·끝, 비율 있음·없음 12칸),
  seed → fill 왕복 동일, mutant 4개 잡힘

기대:
- **prefill_fill_ms**: p1k 3.7 s / p2k 7.2 s → 수백 ms 이하. c-int8 prefill TPS가 c-d14 근처 또는 그 이상
  (c-d14는 chunk마다 fp16 전체 행을 다시 양자화하므로).
- **c-d14와 한 블록 prompt**: 코드 경로가 안 바뀜 → 16회차와 ids·step 줄 같아야 함.

## ① 측정 시킬 때 (그대로 붙여넣기)

```text
nntrainer HTP(Gemma4-26B-A4B, S26U) 17회차: fill HVX화. 코드 수정 없이 측정·보고만.
빌드: fork Seunghui98/nntrainer, 브랜치 htp/first-token-kv-int8. git fetch 후 git rev-parse --short HEAD 가 d283f77b6 인지 확인
(다르면 측정하지 말고 해시만 보고). 6d89d157f 뒤로 바뀐 것: DSP 쪽 nntrainer/tensor/htp_backend/hmx/hexkl_kv_q.c (+ 헤더),
host check, unit test. IDL 그대로.
평소처럼 빌드한 뒤 libnntr_hvx_skel.so와 libnntrainer.so를 새로 올리고 md5를 적을 것 (이번에는 skel이 바뀌었으므로 반드시
올릴 것). libcausallm.so는 16회차(71b2054c…)와 md5가 같으면 그대로, 다르면 새로 올리고 이상한 점에 적을 것.
조건: 16회차와 같음 (R5KL309ALRK, gemma4-26b-a4b-tl2-rotated, C=16, greedy, EOS 끔, 128토큰, max_seq_len 2560,
NNTR_HTP_PAGE_CACHE=0, 매 실행 전 35 ℃ 이하, run.sh).

셀:
  c-int8    NNTR_PREFILL_CHUNK=512
  c-d14     NNTR_PREFILL_CHUNK=512 NNTR_HTP_KV_INT8_ONLY=0
  int8only  (스위치 없음)
  default14 NNTR_HTP_KV_INT8_ONLY=0

## 0. 먼저 확인 (1회)
p2k를 c-int8로 1회 (NNTR_HTP_PROFILE=1). run.sh 요약 줄, pool_sync 줄 원문, 생성 텍스트 앞부분.
- 기대: prefill_seeds=120, prefill_fills=90, prefill_fill_ms가 16회차(7180~7239)보다 크게 작음.
- "FATAL ERROR" / "without seeding its decode cache" / "fell off the int8 KV path" / "could not seed" /
  "could not take its rows" / "not seeded" / "not filled" / FastRPC·DSP 에러 / crash / hang이 나오면 A, B 하지 말고
  그 메시지 원문과 직전 30줄, 명령만 보고.
- ids가 16회차 c-int8 p2k(0b0f4e528a)와 같은지 (scale이 커지는 prompt라 달라도 됨, 같은지 다른지만 적을 것).
문제없으면 A로 진행.

## A. 정확도 (NNTR_HTP_PROFILE 끔)
13회차 acc/ 의 cpu.ids 그대로. 13개 prompt (a01~a11, p1k, p2k)마다 c-int8 1회:
  NNTR_PPL_DECODE=$Q/<prompt>.cpu.ids → r17/logs/<prompt>__c-int8.log
16회차 r16/logs의 <prompt>__c-d14.log 와 <prompt>__ref.log 를 r17/logs로 복사한 뒤 PC에서:
  python3 -I <AI-Study>/NPU/ppl_eval.py r17/logs c-d14 c-int8
출력 원문 전체. prompt마다: chunk 수, "[PPL] decode tokens=" 줄 원문, step 줄이 16회차 c-int8과 같은지
(같음 / 다른 줄 수와 처음 다른 step), 한 블록 prompt(a01 a02 a05~a09)는 15회차 int8only와도 같은지.
그리고 c-d14를 a03, p2k 두 prompt만 1회씩 다시 돌려(같은 설정) step 줄이 16회차 c-d14와 같은지.

## B. 속도·메모리 (NNTR_HTP_PROFILE=1, NNTR_PPL_DECODE 없이)
p1k · p2k, 4셀 각 3회, 셀 순서 돌아가며 (c-int8 → c-d14 → int8only → default14 → ...).
실행마다: prefill TPS, generation TPS, first_token ms, timeline 줄, pool_sync 줄의 prefill_seeds / prefill_fills /
prefill_fill_ms, peak MiB, peak by phase (load / prefill / decode), ids md5 (16회차 같은 셀과 같은지).
중앙값 표, 같은 rep 차이표 (c-int8 − c-d14 의 prefill TPS · prefill_fill_ms · 단계별 peak),
16회차와 비교표 (c-int8 prefill TPS 153.4 / 156.7, prefill_fill_ms 3677 / 7234; c-d14 230.8 / 297.9;
15회차 int8only 612.0 / 660.9, default14 586.3 / 658.5).
rep 1마다 원문: pool_sync 줄, timeline 줄, "(peak by phase KB: ..." 줄.

## 규칙
- 실제로 돌린 것만 "확인됨". 못 돌린 항목은 못 돌렸다고 적을 것.
- 숫자를 추정하거나 지난 결과로 채우지 말 것.
- 실패하면 고치려 하지 말고 에러와 명령을 그대로 보고할 것. A, B 중 하나가 에러로 멈추면 그 항목만 멈추고
  나머지는 진행하되, 에러 원문과 직전 30줄을 적을 것.
- 요청서와 다르게 한 것이 있으면 "이상한 점"에 이유와 함께 적을 것.
```

## ② 결과 정리시킬 때

```text
17회차 결과를 0, A, B 형식 그대로 정리해줘. 맨 앞에 환경(빌드 해시, 새로 올린 파일과 md5, 기기, 모델, 온도, 날짜),
0은 확인 결과와 원문 줄, A는 prompt별 표(prompt | chunk 수 | c-d14 nll_sum / top1 | c-int8 nll_sum / top1 |
16회차 c-int8과 같음? | 15회차 int8only와 같음?(한 블록만)) + ppl_eval.py 출력 원문 + c-d14 재실행 비교,
B는 실행별 표 + 중앙값 표 + 같은 rep 차이표 + 16회차 비교표, 맨 뒤에 "이상한 점"과 원본 로그 경로.
```

## 결과 해석

- **prefill_fill_ms**가 수백 ms 이하이고 c-int8 prefill TPS가 c-d14 근처면 해결. 아직 느리면 fill 안의 bake(HMX tile 굽기)가
  남은 비용이므로 그쪽을 나눠 잼.
- **A**: 한 블록 prompt는 15회차와 같아야 함 (fill을 안 탐). chunk prompt는 16회차 c-int8과 같거나 반올림 경계만큼 다름.
  verdict는 16회차처럼 undecided 근처면 그대로.
- **c-d14 재실행**이 16회차와 다르면 기기·빌드 쪽 변동이므로 이상한 점에 적을 것.
- **B, int8only / default14**: 15회차와 같은 수준이어야 함 (한 블록 경로는 안 바뀜).
