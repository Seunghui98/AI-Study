# 9회차 측정 요청: background seed + warm-up 조기 시작·마감

## ① 측정 시킬 때 (그대로 붙여넣기)

```text
nntrainer HTP(Gemma4-26B-A4B, S26U) 9회차 측정. 코드 수정 없이 측정·보고만.
빌드: fork Seunghui98/nntrainer, 브랜치 htp/first-token-kv-int8.
먼저 git fetch 후 git rev-parse --short HEAD 가 cf9720327 인지 확인 (8회차 첫 실행처럼 옛 빌드를
재는 일이 없도록). 다르면 측정하지 말고 그 해시만 보고. 해시를 보고서 첫 줄에.
다시 빌드·stage. skel(libnntr_hvx_skel.so)이 바뀌었으니 skel과 libnntrainer.so를 둘 다 새로 올릴 것
(IDL은 8회차와 같음).
기기·모델·조건은 8회차와 같음: R5KL309ALRK, models/gemma4-26b-a4b-tl2-rotated, C=16, greedy,
EOS 끔, 128토큰, max_seq_len 2560, NNTR_HTP_PAGE_CACHE=0, 매 실행 전 35 ℃ 이하. run.sh로.
P=/data/local/tmp/nntrainer/causallm/claude_kv/ppl (8회차의 이 모델용 cpu.ids)

## F. prefill 속도, 첫 토큰, 메모리 (p1k, p2k 각각, 셀마다 5회, NNTR_HTP_PROFILE=1)
셀:
  off           (스위치 없음)
  adopt         NNTR_HTP_KV_ADOPT=1
  adopt_warm16  NNTR_HTP_KV_ADOPT=1 NNTR_MOE_WARM=16
셀 순서는 rep마다 돌아가며 (rep1: off, adopt, adopt_warm16 / rep2: adopt, adopt_warm16, off / ...).
실행마다 표 한 줄:
| prompt | cell | rep | prefill TPS | generation TPS | first_token ms | kv_seed_rpc_ms | prefill_seeds / prefill_seed_ms | graph_init_ms (분해 줄 원문) | warm wait ms (있으면) | token1 us / misses / miss_us | peak MiB | 첫 20 id = off rep1? |
(prefill_seeds, prefill_seed_ms 는 "[HTP] t1: pool_sync_ms=" 줄의 값. 이번 빌드에서 prefill_seed_ms 는
seed를 queue에 넣는 시간만이라 8회차의 72 / 84 ms보다 훨씬 작아야 함)
그리고 prompt × cell마다 prefill TPS, generation TPS, first_token, prefill_seed_ms, graph_init_ms, peak MiB의 중앙값.
rep 1마다 같이 원문:
- "[HTP] mem t1 before graph init" / "after graph init" / "after drop host fc" 줄
- "peak memory:" 줄과 "(honest RssAnon+accel: ..." 줄
- "[HTP] t1: pool_sync_ms=" 줄 전체
adopt / adopt_warm16에서 확인:
- "[HTP] prefill seed: layer ... not seeded" 줄 (없어야 함, 있으면 원문 전부)
- "[HTP] decode_kv_prepare:" 줄 (없어야 함, 있으면 원문)
- "t1 seed pos=... from the masters" 줄 (없어야 함, 있으면 개수와 원문 1줄)
- "could not seed" / "lost its int8 masters" / "NNTR_HTP_KV_ADOPT: the decode caches were registered" 예외 (없어야 함)
- prefill_seeds 값 (30이어야 함)
- 생성 텍스트 앞부분 200자 (정상 / 깨짐)
- logcat 또는 stderr에 FastRPC / DSP 에러, crash, hang (있으면 원문)
adopt_warm16에서 추가로 (rep마다):
- "[HTP] t1: warm queued=... at=..." 줄 원문 (at=prefill_end 이어야 함)
- "[HTP] t1: warm registered=... cancelled=... wait_ms=..." 줄 원문
  (8회차는 registered=416, wait 68–139 ms였음. 이번엔 wait가 수 ms, 대신 cancelled가 0보다 클 수 있음)

## G. 정확도 (p1k, p2k, 각 1회)
NNTR_PPL_DECODE=$P/<prompt>.cpu.ids 로 forced 실행:
  1. off   2. NNTR_HTP_KV_ADOPT=1
각 "[PPL] decode tokens=" 줄 전체, 그리고 2의 nll_sum이 8회차 ADOPT(p1k 42.919559598608757,
p2k 69.232775712752982)와 비트 단위로 같은지 (seed 결과는 같아야 함: 계산 위치만 바뀜).
PC에서: python3 ppl_compare.py <1의 로그> <2의 로그>   출력 전체.

## 규칙
- 실제로 돌린 것만 "확인됨". 못 돌린 항목은 못 돌렸다고 적을 것.
- 숫자를 추정하거나 지난 결과로 채우지 말 것.
- 실패하면 고치려 하지 말고 에러와 명령을 그대로 보고할 것.
- 요청서와 다르게 한 것이 있으면 "이상한 점"에 이유와 함께 적을 것.
```

## ② 결과 정리시킬 때 (측정 세션에)

```text
9회차 결과를 요청서의 F, G 형식 그대로 정리해줘. 숫자는 로그에서 그대로 옮기고, 못 돌린 칸은
"못 돌림: 이유"로. 맨 앞에 0. 환경(빌드 해시, skel·libnntrainer 새로 올렸는지, 기기, 모델, 온도, 날짜),
맨 뒤에 "이상한 점"(에러, 경고, 예상과 다른 것, 요청서와 다르게 한 것)과 원본 로그 경로.
F는 실행별 표 전체 + 중앙값 표, G는 ppl_compare.py 출력 원문 전체.
8회차 중앙값(off / adopt, p1k · p2k)과 나란히 놓은 비교표도 하나:
  prefill TPS 584 / 559 · 657 / 637, first_token 417 / 262 · 504 / 217 ms,
  peak 3628 / 3600 · 3638 / 3612 MiB, prefill_seed_ms – / 72 · – / 84.
```

## 결과에 따른 다음 작업

- **adopt의 prefill TPS가 off에 가까워지고**(8회차 −4.3 / −3.0 % → 거의 0), first_token이 8회차 adopt와 같거나
  낮고, peak가 off 이하, G가 비트 동일 → 겹치기 성공.
- **prefill TPS가 그대로면:** background lane이 비지 않아 seed가 MoE bg job 뒤에서 결국 직렬로 돈 것.
  graph_init_ms가 늘었는지(마지막 층들의 seed를 init이 기다림) 같이 보고 unit 크기나 제출 시점 조정.
- **first_token이 늘면:** graph init이 남은 seed를 기다리는 시간. graph_init 분해 줄로 확인.
- **peak가 늘면:** 동시에 살아 있는 int8 cache가 1층보다 많은 것 (reap 시점 문제).
- **adopt_warm16의 first_token이 adopt보다 낮으면** → warm-up 조기 시작·마감 성공. adopt + warm16을
  기본값 후보로. cancelled가 크면(수백) 읽기 시작을 더 앞당길 여지.
- **adopt_warm16이 여전히 adopt보다 느리면:** wait_ms와 token1 misses / miss_us로 어디서 잃는지 확인.
- **G가 다르면:** background seed와 동기 seed가 다른 결과 → 측정 중단하고 원인 분석.
