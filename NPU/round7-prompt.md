# 7회차 측정 요청 (그대로 붙여넣기)

```text
nntrainer HTP(Gemma4-26B-A4B, S26U) 7회차 측정. 코드 수정 없이 측정·보고만.
빌드: after = 1763aa6fc (fork Seunghui98/nntrainer, 브랜치 htp/first-token-kv-int8). 다시 빌드·stage.
기기·모델·조건은 6회차와 같음: R5KL309ALRK, models/gemma4-26b-a4b-tl2-rotated, C=16, greedy,
EOS 끔, 128토큰, max_seq_len 2560, NNTR_HTP_PAGE_CACHE=0, 매 실행 전 35 ℃ 이하.
실행은 6회차 D2처럼 run.sh로.

## E. 6회차 로그에서 다시 읽기 (새 실행 없음)
r6n의 A1 로그 두 개(off, KDIM, p2k, TRACE)에서:
- "HTP attn trace q2fc:" 줄과 "HTP attn trace q2f:" 줄 각각의 개수
- 그 줄들의 필드별 합: append, quant, stage, bake, calib, kernel, qk, softmax, pv, wait, total
- "q2 scales grew" 줄 개수
| cell | q2fc 줄 | q2f 줄 | append | quant | stage | bake | calib | kernel | qk | softmax | pv | wait | total | scales grew |

## F. warm-up 재측정 (p1k)
각 5회 (NNTR_HTP_PROFILE=1): off, NNTR_MOE_WARM=8, NNTR_MOE_WARM=16
실행마다:
- "[HTP] t1:" 줄 전부 (warm queued= / warm registered= 줄이 이번엔 있어야 함, 없으면 그 자체를 보고)
- "token driver: token 1~4" 줄
- first_token ms, peak MiB, 128개 id가 off rep 1과 같은지 (다르면 처음 갈리는 위치)
| cell | rep | first_token ms | token1 us / misses / miss_us | token2~4 misses | warm queued / queue_ms | warm registered / wait_ms | graph_init_ms | peak MiB | ids = off? |
그리고 cell별 first_token, token1 misses, token1 miss_us의 중앙값.

## G. p2k 확인 (F에서 warm이 동작했을 때만)
각 3회: off, F에서 first_token 중앙값이 더 좋았던 warm 값
같은 표 형식.

## 규칙
- 실제로 돌린 것만 "확인됨". 못 돌린 항목은 못 돌렸다고 적을 것.
- 숫자를 추정하거나 지난 결과로 채우지 말 것.
- 실패하면 고치려 하지 말고 에러와 명령을 그대로 보고할 것.
```

## 결과에 따른 다음 작업

- **warm 줄이 찍히고 token 1 misses가 시뮬레이션만큼 줄며(n=8: 약 −109, n=16: 약 −140), peak가 그대로:** 그 n을 기본값으로.
- **warm 줄은 찍히는데 wait_ms가 커서 first_token이 안 줄면:** 읽기가 graph init 안에 다 안 들어가는 것. n을 줄이거나, 더 일찍(마지막 prefill chunk 중에) 시작.
- **ids가 달라지면:** warm expert가 prefill 형식(QS2)으로 읽혀 decode의 TL2와 수치가 다른 것. decode 형식으로 읽도록 수정.
- **E:** KDIM이 남긴 +51 ms가 calibration의 어느 구간인지(또는 scale 증가 재양자화인지) 판정.
