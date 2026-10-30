# 7회차 측정 요청: warm-up + KV adoption (그대로 붙여넣기)

```text
nntrainer HTP(Gemma4-26B-A4B, S26U) 7회차 측정. 코드 수정 없이 측정·보고만.
빌드: after = fork Seunghui98/nntrainer, 브랜치 htp/first-token-kv-int8 의 최신 head
(git rev-parse --short HEAD 를 보고서 첫 줄에). 다시 빌드·stage. 이번 빌드는 IDL에 entry 2개가
추가돼서 skel(libnntr_hvx_skel.so)과 ARM 쪽 stub을 반드시 같이 새로 만들어 올려야 함.
기기·모델·조건은 6회차와 같음: R5KL309ALRK, models/gemma4-26b-a4b-tl2-rotated, C=16, greedy,
EOS 끔, 128토큰, max_seq_len 2560, NNTR_HTP_PAGE_CACHE=0, 매 실행 전 35 ℃ 이하.
실행은 6회차처럼 run.sh로.
P=/data/local/tmp/nntrainer/causallm/claude_kv/ppl (6회차의 이 모델용 cpu.ids를 그대로 씀)

## E. 6회차 로그에서 다시 읽기 (새 실행 없음)
r6n의 A1 로그 두 개(off, KDIM, p2k, TRACE)에서:
- "HTP attn trace q2fc:" 줄과 "HTP attn trace q2f:" 줄 각각의 개수
- 그 줄들의 필드별 합: append, quant, stage, bake, calib, kernel, qk, softmax, pv, wait, total
- "q2 scales grew" 줄 개수
| cell | q2fc 줄 | q2f 줄 | append | quant | stage | bake | calib | kernel | qk | softmax | pv | wait | total | scales grew |

## F. 첫 토큰 셀 (p1k, p2k 각각, 셀마다 5회, NNTR_HTP_PROFILE=1)
셀:
  off           (스위치 없음)
  kdim          NNTR_HTP_Q2_KDIM=1
  adopt         NNTR_HTP_KV_ADOPT=1            (KDIM을 자동으로 켬)
  warm16        NNTR_MOE_WARM=16
  adopt_warm16  NNTR_HTP_KV_ADOPT=1 NNTR_MOE_WARM=16
실행마다 표 한 줄:
| prompt | cell | rep | prefill TPS | first_token ms | kv_seed_rpc_ms | graph_init_ms (분해 줄 원문) | warm queued / registered | token1 us / misses / miss_us | token2~4 misses | peak MiB | 첫 20 id = off rep1? |
그리고 prompt × cell마다 prefill TPS, first_token, kv_seed_rpc_ms, token1 misses, peak MiB의 중앙값.
adopt 셀에서 같이 확인해서 보고:
- "mha_core: t1 seed pos=... from the masters" 줄 (있어야 함, 마지막 seed_ms_sum 값)
- "attn_m1_seed_kvq_tiles: layer ... not seeded" 줄 (없어야 함, 있으면 원문 전부)
- "could not seed its decode cache" / "lost its int8 masters" 예외 (없어야 함, 있으면 원문과 그 직전 30줄)
- "[HTP] t1: pool_sync_ms=" 줄의 "kvq freed late" 값
- 생성 텍스트 앞부분이 정상인지 (깨졌는지)

## G. adopt 정확도 (p1k, p2k, 각 1회)
NNTR_PPL_DECODE=$P/<prompt>.cpu.ids 로 forced 실행:
  1. off   2. NNTR_HTP_Q2_KDIM=1   3. NNTR_HTP_KV_ADOPT=1   4. NNTR_HTP_Q2_KDIM=1 NNTR_HTP_KV_SEED_Q=1
각 "[PPL] decode tokens=" 줄 전체. 그리고 PC에서:
  python3 ppl_compare.py <1의 로그> <2의 로그> <3의 로그> <4의 로그>
출력 전체 (verdict 줄 포함).

## 규칙
- 실제로 돌린 것만 "확인됨". 못 돌린 항목은 못 돌렸다고 적을 것.
- 숫자를 추정하거나 지난 결과로 채우지 말 것.
- 실패하면 고치려 하지 말고 에러와 명령을 그대로 보고할 것.
- 요청서와 다르게 한 것이 있으면 "이상한 점"에 이유와 함께 적을 것.
```

## 결과에 따른 다음 작업

- **adopt:** first_token에서 kv_seed_rpc_ms(96/157 ms)와 graph_init의 attn(55~62 ms)이 줄고, peak가 그대로이거나 줄고, 텍스트가 정상이고, G에서 3/1이 "B worse"가 아니면 → 기본값 후보. 그다음 B 묶음(prefill 중 host fp16 제거) 시작.
- **adopt에서 예외나 깨진 텍스트:** 그 로그로 원인 찾기 (seed shift, 창 크기, 등록 순서).
- **warm:** warm 줄이 찍히고 token1 misses가 시뮬레이션만큼 줄면(n=16: 약 −140) → 기본값. wait_ms가 커서 first_token이 안 줄면 n을 줄이거나 더 일찍 시작.
- **E:** KDIM calib의 남은 +51 ms가 어느 구간인지.
