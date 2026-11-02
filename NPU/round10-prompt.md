# 10회차 측정 요청: decode cache를 한 층 앞서 할당 + seed 시간 분해

## ① 측정 시킬 때 (그대로 붙여넣기)

```text
nntrainer HTP(Gemma4-26B-A4B, S26U) 10회차 측정. 코드 수정 없이 측정·보고만.
빌드: fork Seunghui98/nntrainer, 브랜치 htp/first-token-kv-int8.
먼저 git fetch 후 git rev-parse --short HEAD 가 7d7d28237 인지 확인. 다르면 측정하지 말고 그 해시만 보고.
해시를 보고서 첫 줄에.
다시 빌드·stage. skel(libnntr_hvx_skel.so)과 libnntrainer.so 둘 다 새로 올릴 것 (IDL은 9회차와 같음).
기기·모델·조건은 9회차와 같음: R5KL309ALRK (ANDROID_SERIAL로 지정), models/gemma4-26b-a4b-tl2-rotated,
C=16, greedy, EOS 끔, 128토큰, max_seq_len 2560, NNTR_HTP_PAGE_CACHE=0, 매 실행 전 35 ℃ 이하. run.sh로.
P=/data/local/tmp/nntrainer/causallm/claude_kv/ppl (9회차의 이 모델용 cpu.ids)

## F. prefill 속도, 첫 토큰, 메모리 (p1k, p2k 각각, 셀마다 5회, NNTR_HTP_PROFILE=1)
셀:
  off           (스위치 없음)
  adopt         NNTR_HTP_KV_ADOPT=1
  adopt_warm16  NNTR_HTP_KV_ADOPT=1 NNTR_MOE_WARM=16
셀 순서는 rep마다 돌아가며 (9회차와 같은 순환).
실행마다 표 한 줄:
| prompt | cell | rep | prefill TPS | generation TPS | first_token ms | prefill_seeds / prefill_seed_ms | seed dsp_ms | alloc inline (ms) / ahead (ms) | graph_init_ms (분해 줄 원문) | warm registered / cancelled / wait_ms | token1 us / misses / miss_us | peak MiB | ids md5 = 9회차 같은 셀? |
(seed dsp_ms, alloc 칸은 새 줄 "[HTP] t1: prefill_seed dsp_ms=... alloc inline=N (x ms) ahead=M (y ms)"의 값.
 adopt 셀에서는 inline=2, ahead=28이 기대값)
그리고 prompt × cell마다 prefill TPS, first_token, prefill_seed_ms, seed dsp_ms, peak MiB의 중앙값.
같은 rep끼리 prefill TPS 차이표: (adopt − off), (adopt_warm16 − off), p1k · p2k 각 rep마다.
rep 1마다 같이 원문:
- "[HTP] mem t1 before graph init" / "after graph init" 줄
- "peak memory:" 줄과 "(honest RssAnon+accel: ..." 줄
- "[HTP] t1: pool_sync_ms=" 줄과 바로 다음 "[HTP] t1: prefill_seed dsp_ms=" 줄
adopt / adopt_warm16에서 확인 (모든 rep):
- "[HTP] prefill seed: layer ... not seeded" / "[HTP] decode_kv_prepare:" / "from the masters" 줄 (없어야 함)
- "could not seed" / "lost its int8 masters" 예외 (없어야 함)
- prefill_seeds 값 (30이어야 함)
- 생성 텍스트 앞부분 200자 (정상 / 깨짐)
- logcat 또는 stderr에 FastRPC / DSP 에러, crash, hang (9회차의 init·종료 7줄 외에 있으면 원문)

## G. 정확도 (p1k, p2k, 각 1회)
NNTR_PPL_DECODE=$P/<prompt>.cpu.ids 로 forced 실행:
  1. off   2. NNTR_HTP_KV_ADOPT=1
각 "[PPL] decode tokens=" 줄 전체, 2의 nll_sum이 9회차 ADOPT(p1k 42.919559598608757,
p2k 69.232775712752982)와 비트 단위로 같은지 (할당 시점만 바뀜: 같아야 함).
PC에서: python3 ppl_compare.py <1의 로그> <2의 로그>   출력 전체.

## 규칙
- 실제로 돌린 것만 "확인됨". 못 돌린 항목은 못 돌렸다고 적을 것.
- 숫자를 추정하거나 지난 결과로 채우지 말 것.
- 실패하면 고치려 하지 말고 에러와 명령을 그대로 보고할 것.
- 요청서와 다르게 한 것이 있으면 "이상한 점"에 이유와 함께 적을 것.
```

## ② 결과 정리시킬 때 (측정 세션에)

```text
10회차 결과를 요청서의 F, G 형식 그대로 정리해줘. 숫자는 로그에서 그대로 옮기고, 못 돌린 칸은
"못 돌림: 이유"로. 맨 앞에 0. 환경(빌드 해시, skel·libnntrainer 새로 올렸는지, 기기, 모델, 온도, 날짜),
맨 뒤에 "이상한 점"(에러, 경고, 예상과 다른 것, 요청서와 다르게 한 것)과 원본 로그 경로.
F는 실행별 표 전체 + 중앙값 표 + 같은 rep 차이표, G는 ppl_compare.py 출력 원문 전체.
9회차 중앙값과 나란히 놓은 비교표도 하나 (off / adopt / adopt_warm16, p1k · p2k):
  prefill TPS 596.0 / 607.9 / 567.7 · 655.4 / 641.9 / 641.9,
  first_token 439.2 / 208.2 / 179.5 · 450.9 / 214.1 / 180.0 ms,
  peak 3626 / 3604 / 3607 · 3636 / 3614 / 3614 MiB, prefill_seed_ms – / 52.6 / 56.4 · – / 66.6 / 64.8.
```

## 결과에 따른 다음 작업

- **prefill_seed_ms가 크게 줄고**(ahead=28) 같은 rep의 adopt − off가 0 근처, peak가 off 이하, G 비트 동일
  → adopt + warm16 기본값으로.
- **ahead=28인데 prefill_seed_ms가 그대로면:** 할당이 원인이 아니었던 것. seed dsp_ms가 작으면 RPC 왕복
  (30회) 비용 → seed를 층의 attention 호출에 실어 RPC를 없애는 쪽(IDL 변경).
- **seed dsp_ms가 크면:** DSP 안의 다른 부분(reap, scale 복사). 줄의 inline / ahead 시간과 함께 보고 재분석.
- **ahead가 28보다 작으면:** 할당 스레드가 다음 seed 전에 끝내지 못한 것 (inline 시간이 그 몫).
- **peak가 늘면:** 한 층 앞 할당이 아니라 더 많이 잡히는 것 → 원인 분석.
