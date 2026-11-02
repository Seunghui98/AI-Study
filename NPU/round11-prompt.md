# 11회차 측정 요청: 기본값(adopt + warm16) 확인, params ION, 첫 토큰 분해, 단계별 peak

## ① 측정 시킬 때 (그대로 붙여넣기)

```text
nntrainer HTP(Gemma4-26B-A4B, S26U) 11회차 측정. 코드 수정 없이 측정·보고만.
빌드: fork Seunghui98/nntrainer, 브랜치 htp/first-token-kv-int8.
먼저 git fetch 후 git rev-parse --short HEAD 가 72dd0e1a1 인지 확인. 다르면 측정하지 말고 그 해시만 보고.
해시를 보고서 첫 줄에.
다시 빌드·stage. skel(libnntr_hvx_skel.so)과 libnntrainer.so, libcausallm.so 모두 새로 올릴 것
(IDL은 10회차와 같음).
기기·모델·조건은 10회차와 같음: R5KL309ALRK (ANDROID_SERIAL로 지정), models/gemma4-26b-a4b-tl2-rotated,
C=16, greedy, EOS 끔, 128토큰, max_seq_len 2560, NNTR_HTP_PAGE_CACHE=0, 매 실행 전 35 ℃ 이하. run.sh로.
P=/data/local/tmp/nntrainer/causallm/claude_kv/ppl (10회차의 이 모델용 cpu.ids)

이번 빌드부터 KV adoption과 warm16이 기본으로 켜져 있음 (스위치 없이). 끄는 법: NNTR_HTP_KV_ADOPT=0, NNTR_MOE_WARM=0.

## F. (p1k, p2k 각각, 셀마다 5회, NNTR_HTP_PROFILE=1, 셀 순서는 rep마다 돌아가며)
셀:
  default       (스위치 없음: 이제 adopt + warm16)
  off           NNTR_HTP_KV_ADOPT=0 NNTR_MOE_WARM=0   (10회차 off와 같은 동작)
  ion_off       NNTR_HTP_PARAM_ION=0                   (default에서 params ION만 끔)
실행마다 표 한 줄:
| prompt | cell | rep | prefill TPS | generation TPS | first_token ms | graph_init_ms (분해 줄 원문) | params 줄 (params= mib= staged_ion_mib= ms=) | first_token split (setup / bind / forward / generate / output ms) | pool_sync_ms | token1 us / misses / miss_us | warm registered / cancelled | peak MiB | peak by phase MiB (load / prefill / first_token / decode) | ids md5 = 10회차 같은 동작 셀? |
(새 줄: "[HTP] t1: graph_init params=... mib=... staged_ion_mib=... ms=...",
 "[perf] first_token split: setup_ms=... bind_ms=... forward_ms=... generate_ms=... output_ms=...",
 "(peak by phase KB: load ..., prefill ..., first_token ..., decode ...)" — peak memory 줄 바로 아래)
(ids 비교: default는 10회차 adopt_warm16, off와 ion_off는 각각 10회차 off / adopt_warm16과 같아야 함)
prompt × cell마다 prefill TPS, generation TPS, first_token, graph_init_ms, params ms, peak MiB, 단계별 peak의 중앙값.
같은 rep끼리 차이표: (default − off)의 prefill TPS · first_token · peak, (ion_off − default)의 params ms · first_token.
rep 1마다 원문:
- "[HTP] mem t1 before graph init" / "after graph init" 줄
- "peak memory:" 줄, "(honest RssAnon+accel: ..." 줄, "(peak by phase KB: ..." 줄
- "[HTP] pool: ..." 줄 전부 (특히 "+N experts a layer from the second token", "room for N more a layer")
- "[HTP] t1: pool_sync_ms=" 줄, "prefill_seed dsp_ms=" 줄, "warm queued/registered" 줄
default / ion_off에서 확인 (모든 rep):
- "not seeded" / "decode_kv_prepare:" / "from the masters" / "could not seed" / "lost its int8 masters" (없어야 함)
- prefill_seeds 30, 생성 텍스트 앞 200자 정상 여부, FastRPC / DSP 에러·crash·hang (init·종료 7줄 외)

## G. 정확도 (p1k, p2k, 각 1회)
NNTR_PPL_DECODE=$P/<prompt>.cpu.ids 로 forced 실행:
  1. NNTR_HTP_KV_ADOPT=0 NNTR_MOE_WARM=0   2. 스위치 없음 (default)
각 "[PPL] decode tokens=" 줄 전체. 1의 nll_sum이 10회차 off(p1k 42.541512440801043, p2k 69.254655937952563)와,
2가 10회차 ADOPT(p1k 42.919559598608757, p2k 69.232775712752982)와 비트 단위로 같은지.
PC에서: python3 ppl_compare.py <1의 로그> <2의 로그>   출력 전체.

## 규칙
- 실제로 돌린 것만 "확인됨". 못 돌린 항목은 못 돌렸다고 적을 것.
- 숫자를 추정하거나 지난 결과로 채우지 말 것.
- 실패하면 고치려 하지 말고 에러와 명령을 그대로 보고할 것.
- 요청서와 다르게 한 것이 있으면 "이상한 점"에 이유와 함께 적을 것.
```

## ② 결과 정리시킬 때 (측정 세션에)

```text
11회차 결과를 요청서의 F, G 형식 그대로 정리해줘. 숫자는 로그에서 그대로 옮기고, 못 돌린 칸은
"못 돌림: 이유"로. 맨 앞에 0. 환경(빌드 해시, skel·libnntrainer·libcausallm 새로 올렸는지, 기기, 모델, 온도, 날짜),
맨 뒤에 "이상한 점"(에러, 경고, 예상과 다른 것, 요청서와 다르게 한 것)과 원본 로그 경로.
F는 실행별 표 전체 + 중앙값 표 + 같은 rep 차이표, G는 ppl_compare.py 출력 원문 전체.
10회차 중앙값과 나란히 놓은 비교표도 하나 (default ↔ 10회차 adopt_warm16, off ↔ 10회차 off):
  prefill TPS 605.6 / 597.9 · 660.5 / 663.5, first_token 174.3 / 436.1 · 183.0 / 500.9 ms,
  peak 3606 / 3626 · 3616 / 3636 MiB, graph_init params 38.2 / 45.0 · 38.7 / 44.7 ms (rep1).
```

## 결과에 따른 다음 작업

- **default가 10회차 adopt_warm16과 같은 숫자·ids** → 기본값 전환 확인.
- **params ms가 ion_off보다 크게 작으면** → ION staging 유지. 비슷하면 FastRPC 복사가 원인이 아님 → router 전송 방식 재검토.
- **first_token split:** forward_ms가 graph_init + pool_sync + token1과 맞는지, 나머지(setup / generate / output)가 ~50 ms의 어디인지.
- **peak by phase:** 어느 단계가 peak를 정하는지. decode가 peak이고 off와 default의 decode peak가 비슷하면 → peak는 pool
  성장(PD 매핑 상한)이 정하는 것이라 prefill 중 host fp16 제거(C 구조 2단계)는 peak를 줄이지 못함
  (대신 pool이 더 커져 decode가 빨라짐). prefill이나 load가 peak면 → C 구조 2단계가 peak를 직접 줄임.
