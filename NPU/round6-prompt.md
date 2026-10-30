# 6회차 측정 요청 (그대로 붙여넣기)

```text
nntrainer HTP(Gemma4-26B-A4B, S26U v81) 6회차 측정. 코드 수정 없이 측정·보고만.
빌드: after = f407d991e (fork Seunghui98/nntrainer, 브랜치 htp/first-token-kv-int8). 다시 빌드·stage.
조건은 4회차와 같음 (C=16, greedy, EOS 끔, 128토큰, max_seq_len 2560,
NNTR_HTP_PAGE_CACHE=0, 배터리 35 ℃ 이하 시작). 원래 pack (fp16 pack 아님).
P=/data/local/tmp/nntrainer/causallm/claude_kv/ppl
분석 스크립트: AI-Study 저장소 NPU/ppl_compare.py, NPU/expert_warm_sim.py (PC에서 python3로 실행)

## A. KDIM calibration 시간 (p2k)
A1. NNTR_HTP_PROFILE=1 NNTR_HTP_ATTN_TRACE=1 로 각 1회: off, NNTR_HTP_Q2_KDIM=1
    각각: 30층 "calib=" 합, "accel_call_us" 합, q2 clip 줄 개수.
    2회차 기준: off 60~62 ms, KDIM 133 ms.
A2. TRACE 없이 각 3회: off, NNTR_HTP_Q2_KDIM=1
    prefill TPS 중앙값, 첫 20개 id가 off와 같은지.
    2회차 기준: KDIM prefill이 off보다 -3.6 %.

## B. DSP 경로 대 NEON 경로 (p1k, 1회)
NNTR_HTP_Q2_KDIM=1 NNTR_HTP_ATTN_F32_CHECK=1 NNTR_HTP_ATTN_TRACE=1
- "f32 edge check" 줄의 mismatches= 값 분포 (층별, 0이 몇 층인지)
- 같은 것을 스위치 off로도 1회 (대조)

## C. KDIM nll 판정 (p1k, p2k). 4회차의 $P/<prompt>.cpu.ids를 그대로 씀 (지우거나 다시 쓰지 말 것)
모두 NNTR_PPL_DECODE=$P/<prompt>.cpu.ids 로 forced 실행, 로그 전체 보관:
1. off
2. NNTR_HTP_Q2_KDIM=1
3. NNTR_HTP_Q2_KDIM=1 NNTR_HTP_ATTN_DSP_CALIB=0
4. NNTR_HTP_ATTN_DSP_CALIB=0
5. NNTR_HTP_Q2_KDIM=1 NNTR_HTP_KV_SEED_Q=1   (노이즈 기준)
각각 "[PPL] decode tokens=" 줄 전체, "[PPL] decode warning" 유무.
그리고 PC에서:
  python3 ppl_compare.py <1의 로그> <2의 로그> <5의 로그>
  python3 ppl_compare.py <4의 로그> <3의 로그>
출력 전체를 붙여줄 것 (verdict 줄 포함).
1과 2의 nll_sum이 4회차 off / KDIM과 같은지도 비교.

## D. 첫 토큰 (p1k)
D1. 스위치 off 1회, 라우팅 로그:
    NNTR_HTP_PROFILE=1 NNTR_MOE_TRACE=$P/p1k.moe.txt NNTR_HTP_ROUTE_LOG=$P/p1k.route.txt
    두 파일을 PC로 가져와: python3 expert_warm_sim.py p1k.moe.txt p1k.route.txt
    출력 표 전체.
D2. 각 3회 (NNTR_HTP_PROFILE=1): off, NNTR_MOE_WARM=8, NNTR_MOE_WARM=16
    각각: "[HTP] t1:" 줄 전부 (graph_init_ms 분해, warm queued / registered 줄 포함),
    "token driver: token 1~4" 줄, first_token ms, peak memory, 첫 20개 id가 off와 같은지.
    (run_first_token_tests.sh: CASES_ONLY="base warm8 warm16" bash run_first_token_tests.sh run p1024.txt)

## 규칙
- 실제로 돌린 것만 "확인됨". 못 돌린 항목은 못 돌렸다고 적을 것.
- 숫자를 추정하거나 지난 결과로 채우지 말 것.
- 실패하면 고치려 하지 말고 에러와 명령을 그대로 보고할 것.
```

## 결과에 따른 다음 작업

- **B에서 KDIM의 mismatch가 off 수준이고, C의 2/1 verdict가 "B not worse"이거나 undecided:** KDIM 통과로 보고 C 구조 S1부터 구현.
- **C의 2/1이 "B worse"이고 3/4는 아님:** DSP calibration 쪽 버그. B의 mismatch 위치로 좁힘.
- **2/1, 3/4 모두 "B worse":** KDIM 자체의 손해. C 구조 중단, KDIM·SEED_Q 정리.
- **D1의 tok1 hits가 n=8에서 의미 있게 크고 D2에서 token 1 miss·first_token이 줄며 peak가 그대로:** warm-up 기본값으로.
- **D2의 graph_init 분해:** 메모리를 늘리지 않는 구간(rpc, params, e2e 등)이 크면 prefill 앞으로 옮김.
