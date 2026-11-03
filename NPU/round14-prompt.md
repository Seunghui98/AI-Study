# 14회차 측정 요청: 정확도 재판정(기존 로그) + 첫 토큰 확인

## ① 측정 시킬 때 (그대로 붙여넣기)

```text
nntrainer HTP(Gemma4-26B-A4B, S26U) 14회차. 코드 수정 없이 측정·보고만.

## A. 정확도 재판정 (기기 실행 없음, 13회차 로그 그대로)
AI-Study 저장소를 pull (ppl_eval.py의 sign test 오버플로를 고친 판). 그리고:
  python3 -I <AI-Study>/NPU/ppl_eval.py ~/workspace/kv-int8-device/r13/logs off default
출력 원문 전체 (pooled 줄, sign test, margin/gap 요약 두 줄, verdict 줄까지).

## B. 첫 토큰 확인
빌드: fork Seunghui98/nntrainer, 브랜치 htp/first-token-kv-int8. git fetch 후 git rev-parse --short HEAD 가
7cc5c0149 인지 확인 (다르면 측정하지 말고 해시만 보고). IDL이 12회차 것으로 되돌아감(graph_set_params 제거) →
13회차처럼 IDL에서 stub/skel을 다시 생성하는 전체 빌드로 skel, libnntrainer.so, libcausallm.so 새로 올릴 것.
조건은 13회차 4.와 같음 (R5KL309ALRK, C=16, greedy, EOS 끔, 128토큰, max_seq_len 2560, NNTR_HTP_PAGE_CACHE=0,
35 ℃ 이하, run.sh, NNTR_HTP_PROFILE=1, NNTR_PPL_DECODE 없이). p1k · p2k, 셀마다 5회, 셀 순서 돌아가며:
  default (스위치 없음)   off (NNTR_HTP_KV_ADOPT=0 NNTR_MOE_WARM=0)
실행마다: prefill TPS, generation TPS, first_token ms, timeline 줄 (graph_init / to_token / pool_sync / to_packet /
packet / cleanup), params 줄 (params / mib / ms / dsp_ms), token1 us / misses, warm registered / cancelled, peak MiB,
peak by phase, ids md5가 13회차 같은 셀과 같은지.
중앙값 표, 같은 rep 차이표 (default − off의 first_token · peak), 12·13회차와 비교표:
  first_token 12회차 170.3 / 427.8 · 175.5 / 498.6, 13회차 172.1 / 408.9 · 176.8 / 491.7 ms
  to_token 12회차 35.4 / 138.1 · 39.1 / 209.3, 13회차 17.8 / 123.2 · 18.8 / 186.1 ms
  params 12회차 38.3 / 43.9 · 38.8 / 44.7, 13회차 53.7 / 59.8 · 60.2 / 65.3 ms
  peak 12회차 3606 / 3626 · 3615 / 3636, 13회차 3613 / 3633 · 3624 / 3643 MiB
rep 1마다 원문: timeline 줄, params 줄, "peak memory:" / "(honest ..." / "(peak by phase KB: ..." 줄.
default에서 확인: "not seeded" / "could not seed" / "lost its int8 masters" 없음, prefill_seeds 30, 텍스트 정상,
FastRPC / DSP 에러·crash·hang (init·종료 7줄 외).

## 규칙
- 실제로 돌린 것만 "확인됨". 못 돌린 항목은 못 돌렸다고 적을 것.
- 숫자를 추정하거나 지난 결과로 채우지 말 것.
- 실패하면 고치려 하지 말고 에러와 명령을 그대로 보고할 것.
- 요청서와 다르게 한 것이 있으면 "이상한 점"에 이유와 함께 적을 것.
```

## ② 결과 정리시킬 때

```text
14회차 결과를 A, B 형식 그대로 정리해줘. 맨 앞에 0. 환경(빌드 해시, 새로 올린 파일과 md5, 기기, 모델, 온도, 날짜),
A는 ppl_eval.py 출력 원문 전체, B는 실행별 표 + 중앙값 표 + 같은 rep 차이표 + 12·13회차 비교표,
맨 뒤에 "이상한 점"과 원본 로그 경로.
```

## 결과 해석

- B: params가 12회차 수준(38 ms)으로, to_token이 13회차 수준(18 ms)으로 → 첫 토큰 약 155 ms 기대. peak는 12회차 수준.
- A: verdict와 margin 요약. default가 기준과 다르게 고른 단계에서 기준의 margin이 대부분 0.5 미만이면 동점 근처의 뒤집힘.
