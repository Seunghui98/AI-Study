# 12회차 측정 요청: 첫 토큰 정리 작업 뒤로 미루기 + 첫 토큰 timeline + params의 DSP 시간

## ① 측정 시킬 때 (그대로 붙여넣기)

```text
nntrainer HTP(Gemma4-26B-A4B, S26U) 12회차 측정. 코드 수정 없이 측정·보고만.
빌드: fork Seunghui98/nntrainer, 브랜치 htp/first-token-kv-int8.
먼저 git fetch 후 git rev-parse --short HEAD 가 551f3e0bc 인지 확인. 다르면 측정하지 말고 그 해시만 보고.
해시를 보고서 첫 줄에.
다시 빌드·stage. skel(libnntr_hvx_skel.so)과 libnntrainer.so, libcausallm.so 모두 새로 올릴 것
(IDL은 11회차와 같음). stage.sh가 ANDROID_NDK 없음으로 libc++_shared 단계에서 멈추면 11회차처럼
앞 단계 파일들의 md5만 확인하고 진행 (libc++_shared.so는 기기 것 그대로).
기기·모델·조건은 11회차와 같음: R5KL309ALRK (ANDROID_SERIAL로 지정), models/gemma4-26b-a4b-tl2-rotated,
C=16, greedy, EOS 끔, 128토큰, max_seq_len 2560, NNTR_HTP_PAGE_CACHE=0, 매 실행 전 35 ℃ 이하. run.sh로.
P=/data/local/tmp/nntrainer/causallm/claude_kv/ppl

## F. (p1k, p2k 각각, 셀마다 5회, NNTR_HTP_PROFILE=1, 셀 순서는 rep마다 돌아가며)
셀:
  default    (스위치 없음: adopt + warm16, 첫 토큰 정리 작업을 둘째 토큰으로 미룸)
  nodefer    NNTR_HTP_T1_DEFER=0                    (정리 작업을 11회차처럼 첫 토큰 안에서)
  off        NNTR_HTP_KV_ADOPT=0 NNTR_MOE_WARM=0
실행마다 표 한 줄:
| prompt | cell | rep | prefill TPS | generation TPS | first_token ms | 2번째 토큰 ms (있으면) | timeline 줄 (graph_init / to_token / pool_sync / to_packet / packet / cleanup) | after-token cleanups ms | params 줄 (params / mib / ms / dsp_ms) | token1 us / misses / miss_us | warm registered / cancelled | peak MiB | peak by phase MiB (load / prefill / first_token / decode) | ids md5 = 11회차 같은 동작 셀? |
(새 줄: "[HTP] t1: timeline_ms graph_init=... to_token=... pool_sync=... to_packet=... packet=... cleanup=..."
 (default에서는 끝에 "(deferred to the second token)"), "[HTP] t1: after-token cleanups ms=...",
 params 줄 끝의 "dsp_ms=...". 2번째 토큰 ms는 로그에 step별 시간이 있으면 그 값, 없으면 "없음")
(ids: default와 nodefer는 11회차 default, off는 11회차 off와 같아야 함)
prompt × cell마다 prefill TPS, generation TPS, first_token, params ms, params dsp_ms, peak MiB, 단계별 peak의 중앙값.
같은 rep끼리 차이표: (default − nodefer)의 first_token · generation TPS, (default − off)의 first_token · peak.
rep 1마다 원문: "peak memory:" 줄, "(honest ..." 줄, "(peak by phase KB: ..." 줄, timeline 줄, cleanups 줄, params 줄,
 "[HTP] pool: ..." 줄 전부, "[HTP] token driver: first token" 줄.
default / nodefer에서 확인 (모든 rep): "not seeded" / "could not seed" / "lost its int8 masters" 없음, prefill_seeds 30,
 생성 텍스트 앞 200자 정상, FastRPC / DSP 에러·crash·hang (init·종료 7줄 외).

## G. 정확도 (p1k, p2k, 각 1회)
NNTR_PPL_DECODE=$P/<prompt>.cpu.ids 로 forced 실행: 1. NNTR_HTP_KV_ADOPT=0 NNTR_MOE_WARM=0  2. default
각 "[PPL] decode tokens=" 줄 전체, 11회차와 비트 동일한지 (1: 42.541512440801043 / 69.254655937952563,
2: 42.919559598608757 / 69.232775712752982). PC에서 python3 ppl_compare.py <1> <2> 출력 전체.

## 규칙
- 실제로 돌린 것만 "확인됨". 못 돌린 항목은 못 돌렸다고 적을 것.
- 숫자를 추정하거나 지난 결과로 채우지 말 것.
- 실패하면 고치려 하지 말고 에러와 명령을 그대로 보고할 것.
- 요청서와 다르게 한 것이 있으면 "이상한 점"에 이유와 함께 적을 것.
```

## ② 결과 정리시킬 때 (측정 세션에)

```text
12회차 결과를 요청서의 F, G 형식 그대로 정리해줘. 숫자는 로그에서 그대로 옮기고, 못 돌린 칸은
"못 돌림: 이유"로. 맨 앞에 0. 환경(빌드 해시, skel·libnntrainer·libcausallm 새로 올렸는지, 기기, 모델, 온도, 날짜),
맨 뒤에 "이상한 점"(에러, 경고, 예상과 다른 것, 요청서와 다르게 한 것)과 원본 로그 경로.
F는 실행별 표 전체 + 중앙값 표 + 같은 rep 차이표, G는 ppl_compare.py 출력 원문 전체.
11회차 중앙값과 나란히 놓은 비교표 (default / off, p1k · p2k):
  first_token 177.5 / 435.6 · 186.0 / 502.0 ms, generation TPS 17.40 / 16.91 · 16.48 / 16.03,
  peak 3606 / 3626 · 3616 / 3636 MiB, params ms 40.5 / 46.0 · 40.4 / 45.5.
```

## 결과에 따른 다음 작업

- **default의 first_token이 nodefer보다 cleanup ms만큼 작고** generation TPS가 같으면 → 미루기 유지.
- **timeline:** to_token(graph init 끝 → 첫 토큰 packet 진입)이 크면 모델 쪽 host 작업, to_packet이 크면 packet 준비.
  남은 ~50 ms가 어디인지로 다음 대상을 정함.
- **params dsp_ms:** ms에 가까우면 DSP 쪽 처리(router 포장)가 원인 → 그 처리 개선. 작으면 301번 RPC 왕복
  → 한 번의 RPC로 묶기(IDL 변경).
- **peak by phase의 decode 값:** 이번에 처음 제대로 찍힘. peak가 decode 단계면 prefill 쪽 메모리 절감은 peak에 안 보임.
