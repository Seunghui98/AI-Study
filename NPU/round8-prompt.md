# 8회차 측정 요청: prefill 안에서 seed (1안) + KDIM calib 정리

## ① 측정 시킬 때 (그대로 붙여넣기)

```text
nntrainer HTP(Gemma4-26B-A4B, S26U) 8회차 측정. 코드 수정 없이 측정·보고만.
빌드: fork Seunghui98/nntrainer, 브랜치 htp/first-token-kv-int8 의 최신 head
(git rev-parse --short HEAD 를 보고서 첫 줄에). 다시 빌드·stage. skel(libnntr_hvx_skel.so)이 바뀌었으니
skel과 libnntrainer.so를 둘 다 새로 올릴 것 (IDL은 7회차와 같음).
기기·모델·조건은 7회차와 같음: R5KL309ALRK, models/gemma4-26b-a4b-tl2-rotated, C=16, greedy,
EOS 끔, 128토큰, max_seq_len 2560, NNTR_HTP_PAGE_CACHE=0, 매 실행 전 35 ℃ 이하. run.sh로.
P=/data/local/tmp/nntrainer/causallm/claude_kv/ppl (6·7회차의 이 모델용 cpu.ids)

## H. 7회차 로그에서 다시 읽기 (새 실행 없음)
r7 p1k rep 1의 off와 adopt 로그에서 다음 줄 원문:
- "[HTP] mem t1 before graph init", "[HTP] mem t1 after graph init", "[HTP] mem t1 after drop host fc"
- "[HTP] pool: ... room for N more a layer", "[HTP] pool: +N experts a layer from the second token"
- "peak memory:" 줄과 그 다음 "(honest RssAnon+accel: ..." 줄
- "[HTP] attn_m1: registered ..." 줄 전부

## F. 첫 토큰과 메모리 (p1k, p2k 각각, 셀마다 5회, NNTR_HTP_PROFILE=1)
셀:
  off           (스위치 없음)
  kdim          NNTR_HTP_Q2_KDIM=1
  adopt         NNTR_HTP_KV_ADOPT=1
  warm16        NNTR_MOE_WARM=16
  adopt_warm16  NNTR_HTP_KV_ADOPT=1 NNTR_MOE_WARM=16
실행마다 표 한 줄:
| prompt | cell | rep | prefill TPS | generation TPS | first_token ms | kv_seed_rpc_ms | prefill_seeds / prefill_seed_ms | graph_init_ms (분해 줄 원문) | token1 us / misses / miss_us | peak MiB | 첫 20 id = off rep1? |
(prefill_seeds, prefill_seed_ms 는 "[HTP] t1: pool_sync_ms=" 줄에 새로 생긴 값)
그리고 prompt × cell마다 prefill TPS, generation TPS, first_token, kv_seed_rpc_ms, prefill_seed_ms, peak MiB의 중앙값.
rep 1마다 같이 원문:
- "[HTP] mem t1 before graph init" / "after graph init" / "after drop host fc" 줄
- "peak memory:" 줄과 "(honest RssAnon+accel: ..." 줄
- "[HTP] attn_m1: registered ..." 줄 (adopt 셀에서는 prefill 전에 찍혀야 함: 앞뒤 줄 몇 개도 같이)
adopt / adopt_warm16에서 확인:
- "[HTP] prefill seed: layer ... not seeded" 줄 (없어야 함, 있으면 원문 전부)
- "[HTP] decode_kv_prepare:" 줄 (없어야 함, 있으면 원문)
- "t1 seed pos=... from the masters" 줄 (이번엔 없어야 함: seed가 prefill 안에서 끝남. 있으면 개수와 원문 1줄)
- "could not seed" / "lost its int8 masters" / "NNTR_HTP_KV_ADOPT: the decode caches were registered" 예외 (없어야 함)
- pool_sync 줄의 prefill_seeds 값 (30이어야 함)과 "seeds=... (dsp ..." 값
- 생성 텍스트 앞부분 200자 (정상 / 깨짐)

## G. 정확도 (p1k, p2k, 각 1회)
NNTR_PPL_DECODE=$P/<prompt>.cpu.ids 로 forced 실행:
  1. off   2. NNTR_HTP_KV_ADOPT=1   3. NNTR_HTP_Q2_KDIM=1
각 "[PPL] decode tokens=" 줄 전체, 그리고 2와 3의 nll_sum이 7회차의 ADOPT(p1k 42.919559598608757,
p2k 69.232775712752982) / KDIM(p1k 46.110528403409873, p2k 68.103675873970985)과 비트 단위로 같은지.
PC에서: python3 ppl_compare.py <1의 로그> <2의 로그> <3의 로그>   출력 전체.

## E. KDIM calib (p2k, 각 1회, NNTR_HTP_PROFILE=1 NNTR_HTP_ATTN_TRACE=1)
off, NNTR_HTP_Q2_KDIM=1 각각: "HTP attn trace q2fc:" 줄의 calib 합, kernel 합, wall_us 합 (30층)
7회차 E: off calib 61.3 / wall 481.0, KDIM calib 112.8 / wall 554.9 ms

## 규칙
- 실제로 돌린 것만 "확인됨". 못 돌린 항목은 못 돌렸다고 적을 것.
- 숫자를 추정하거나 지난 결과로 채우지 말 것.
- 실패하면 고치려 하지 말고 에러와 명령을 그대로 보고할 것.
- 요청서와 다르게 한 것이 있으면 "이상한 점"에 이유와 함께 적을 것.
```

## ② 결과 정리시킬 때 (측정 세션에)

```text
8회차 결과를 요청서의 H, F, G, E 형식 그대로 정리해줘. 숫자는 로그에서 그대로 옮기고, 못 돌린 칸은
"못 돌림: 이유"로. 맨 앞에 0. 환경(빌드 해시, skel·libnntrainer 새로 올렸는지, 기기, 모델, 온도, 날짜),
맨 뒤에 "이상한 점"(에러, 경고, 예상과 다른 것, 요청서와 다르게 한 것)과 원본 로그 경로.
F는 실행별 표 전체 + 중앙값 표, G는 ppl_compare.py 출력 원문 전체.
```

## 결과에 따른 다음 작업

- **adopt의 peak가 off와 같거나 낮고**, kv_seed_rpc_ms가 거의 0, prefill_seeds=30, 텍스트 정상, G의 2가 7회차 ADOPT와 비트 동일이면 → 1안 성공. adopt + warm16을 기본값 후보로.
- **peak가 여전히 높으면:** H와 F의 mem 줄(s1_heap_mib)로 어느 시점에 커지는지 보고 원인 재분석.
- **prefill TPS가 많이 떨어지면:** prefill_seed_ms가 prefill에 그대로 더해진 것. seed를 다음 층 MoE와 겹치는 작업으로.
- **E:** KDIM calib가 off 수준(약 61 ms)으로 내려왔는지.
