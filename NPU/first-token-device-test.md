# 첫 토큰 / KV int8 — 진행 상태와 기기 측정 기록

코드: `Seunghui98/nntrainer` 브랜치 `htp/first-token-kv-int8` (base `a3640d2`, head `1763aa6`)

| 커밋 | 내용 | 스위치 (기본 off) |
|---|---|---|
| `ecb6eb4` | prefill int8 K scale을 (head, dim)당으로, 비율은 Q에 접음 | `NNTR_HTP_Q2_KDIM=1` |
| `f30bde9` | KDIM일 때 decode seed의 K도 int8 masters에서. seed fallback 로그 | `NNTR_HTP_KV_SEED_Q=1` (기존) |
| `568fa2e` | host NEON u16 Q 경로도 같은 방식으로 접음 | - |
| `d1b6873` | 스위치 off의 DSP calibration을 a3640d2와 같게 되돌림 (`ecb6eb4`의 prefill 회귀 제거). KDIM 경로는 레지스터 누적 | - |
| `ec09f64` | `NNTR_PPL_DECODE` 다시 추가 (decode nll 게이트, skip_prefill 지원) | `NNTR_PPL_DECODE=<ids 파일>` |
| `c2bb62d` | PROFILE에서 decode 토큰 1–4를 한 줄씩 (`token driver: token N ... misses= miss_us=`) | `NNTR_HTP_PROFILE=1` |
| `6f71c9b` | KDIM calibration의 lane 병합을 HVX로 (2회차 calib 62→133 ms의 원인). q2 clip trace의 Q 접기 수정 | - |
| `4294d96` | host 검사 추가: DSP 커널의 KDIM Q 접기(`stage_qprep_kd`)가 host NEON 공식과 tile 바이트 동일 | - |
| `d765641` | 첫 토큰 expert warm-up: prefill 라우팅으로 층별 n개를 graph init 동안 미리 읽음. pool 크기 그대로 | `NNTR_MOE_WARM=<n>` |
| `f407d99` | `t1: graph_init_ms` 줄에 구간 분해 (place_fc / bind / rpc / params / attn / e2e) | - |
| `1763aa6` | warm-up을 Gemma 4가 실제로 쓰는 `Lfm2MoELayer`에도 (6회차에서 hook이 등록되지 않던 버그) | `NNTR_MOE_WARM=<n>` |

스위치를 다 끄면 base와 같다 (2회차에서 ids 비트 동일, prefill 노이즈 범위 안).

---

## 목표와 계획 (C 구조)

prefill과 decode가 decode 형식의 int8 KV 한 벌만 쓴다. host fp16 KV 없음, 첫 decode의 seed 변환 없음.

1. 회귀 제거 — 완료 (`d1b6873`, 2회차 통과)
2. KDIM 정확도 게이트 — 통과 (6회차: DSP=NEON, nll 차이 없음, 5회차 층별 오차는 KDIM이 절반)
3. 6k/22k에서 scale 증가 빈도, peak 위치 측정
4. 구현: masters를 decode 형식으로, prefill RoPE·양자화를 DSP로, 첫 decode가 masters를 넘겨받음, HMX tile은 호출마다 해제, scale 증가는 int8 재양자화(2배 단위), sliding compaction은 DSP memmove, SEED_Q 삭제

예상 (추정): first_token에서 seed 93 / 157 ms(p1k / p2k) 제거, prefill 구간 KV 메모리 약 −260 MiB (max_seq 2560).

KDIM이 게이트에서 떨어지면 C는 성립하지 않음 (head당 K scale로는 decode nll +9.4 %): KDIM·SEED_Q를 정리하고 첫 토큰은 graph_init / poolSync 쪽에서 찾는다.

---

## 1회차 (after = `568fa2e`, PAGE_CACHE 미설정)

- 스위치 off에서도 prefill −4~6 %. 원인은 `ecb6eb4`의 DSP calibration (off에서도 per-dim 메모리 누적). → `d1b6873`
- SEED_Q는 seed RPC 안으로 할당·해제가 옮겨갈 뿐 첫 토큰 단축 없음 (seed의 바이트 이동이 스칼라).

## 2회차 (after = `ec09f64`, `NNTR_HTP_PAGE_CACHE=0`)

1회차와 PAGE_CACHE 조건이 달라 절대값은 회차 안에서만 비교.

| prompt | cell | prefill TPS | first_token ms | kv_seed_rpc ms | peak MiB | ids |
|---|---|---|---|---|---|---|
| p1k | before | 578.2 | 440 | 92.6 | 3630 | 기준 |
| p1k | after off | 577.0 | 437 | 93.4 | 3630 | 같음 |
| p1k | KDIM | 563.6 | 440 | 93.3 | 3631 | 다름 |
| p1k | KDIM + SEED_Q | 577.9 | 408 | 116.8 | 3979 | 다름 |
| p2k | before | 651.8 | 498 | 157.0 | 3641 | 기준 |
| p2k | after off | 649.0 | 506 | 157.0 | 3640 | 같음 |
| p2k | KDIM | 628.3 | 507 | 157.2 | 3641 | 다름 |
| p2k | KDIM + SEED_Q | 627.3 | 441 | 164.9 | 4109 | 다름 |

- 회귀: 해소 (off 대 before, prefill 중앙값 −0.2 / −0.4 %, ids 비트 동일, calib 60.0 대 62.4 ms)
- KDIM: calib 62 → 133 ms (lane 병합이 스칼라), p2k prefill −3.6 %
- SEED_Q: first_token −29 / −65 ms, 대신 peak +349 / +468 MiB (PAGE_CACHE=0에서 `release_prefill`이 꺼져 30층 prefill cache가 남음)
- nll (기준 = off의 self 경로): KDIM 1.108 / 1.156, KDIM+SEED_Q 1.088 / 1.053 → 1.02 게이트 실패. 다만 KDIM과 KDIM+SEED_Q는 decode cache가 tie에서만 다른데도 비율이 0.02~0.10 달라, 이 측정의 노이즈가 게이트보다 큼. 기준이 자기 경로를 채점해 off에 유리함 → 판정 보류

현재 브랜치는 스위치 off로 before와 같고, 켜는 조합은 아직 순이득이 없다.

---

## 3회차 설계: 편향 없는 nll 게이트

- 경로 작성: prefill attention을 fp16으로 (pack의 `attention_kv_dtype`를 `""`로 바꾼 symlink pack), `NNTR_PPL_DECODE` self
- 채점: 원래 pack의 off / KDIM / KDIM+SEED_Q가 같은 파일을 forced
- off와 KDIM은 decode seed가 같고 prefill K scale만 다름 → 같은 경로 위에서 nll이 낮은 쪽이 fp16 prefill에 가까움
- 노이즈 = KDIM+SEED_Q / KDIM (tie 수준 차이)
- 판정: KDIM / off가 노이즈 범위 안이거나 1 미만이면 C 진행, 노이즈보다 분명히 크면 C 중단
- 같이: 2회차 로그의 `[HTP] t1: graph_init_ms=`, `pool_sync_ms=`, `t1 seed`, `token driver: first token` 줄 (첫 토큰 분해)

---

## 3회차 결과: fp16 경로 작성자 실패

fp16 pack(prefill attention fp16)으로 만든 경로가 깨진 텍스트. 원인은 4회차 A1에서 확인: DSP fp16 attention 커널(`attn_f16_prefill`)이 Gemma 4 shape에서 모든 층 rel_rms = nan. 지금 목표와 별개라 기록만. nll 기준은 CPU attention 경로로 바꿈.

## 4회차: CPU attention 경로 기준 nll, 첫 토큰 분해

| 비율 (nll_sum) | p1k | p2k |
|---|---|---|
| KDIM / off | 1.042 | 1.115 |
| 노이즈: KDIM+SEED_Q / KDIM | 0.971 | 1.038 |

p2k에서 KDIM이 노이즈보다 나쁨. 첫 토큰 (off, p1k, `c2bb62d`의 줄):

| token | us | misses | miss_us |
|---|---|---|---|
| 1 | 182370 | 220 | 130127 |
| 2 | 115080 | 178 | 54570 |
| 3 | 92060 | 95 | 36155 |
| 4 | 71083 | 80 | 17284 |

## 5회차: 층별 attention 오차 (`NNTR_HTP_ATTN_F16_CHECK=1`)

host calibration + NEON Q 경로, 층마다 CPU fp16 attention과 비교 (오차 전파 없음). KDIM rel_rms 평균 0.0158 대 off 0.0291 (p1k). KDIM이 더 작은 층 30/30, 29/30 (p1k, p2k). layer 1은 둘 다 약 0.22.

→ 4회차(KDIM 나쁨)와 충돌.

---

## 충돌 분석 (host, 기기 없이)

4회차의 일반 실행은 DSP calibration + DSP 커널 안의 Q 접기, 5회차는 host calibration + NEON Q 접기. KDIM 전용 코드 중 5회차가 지나지 않은 DSP 경로를 하나씩 host에서 확인:

| DSP 경로 | 확인 | 결과 |
|---|---|---|
| calibration lane 워커 (K/V/Q 최대·최소) | `Q2C CALIB WORKERS OK` | 스칼라 참조와 비트 동일 |
| calibration의 접힌 Q 범위 병합 | 같은 검사 (`6f71c9b`) | 교체 전 스칼라 병합과 비트 동일 |
| 커널의 Q 접기 + u16 양자화 (`stage_qprep_kd`) | `Q2 QPREP KD OK` (`4294d96`) | host 공식으로 만든 u16과 tile 바이트 동일, mutant 3개 잡힘 |
| per-dim K append 양자화 | `run_kv_fixed_tail_check`, `unittest_hexkl_kv_q` | 기존 통과 |
| chunk 사이 scale 재전송 | 코드 확인 | kd 플래그 전달됨 (`kv_cache_q_set_fixed_scales`) |
| alpha | 코드 확인 | fs_k(머리당 최대) 사용, 접힌 Q와 맞음 |

host에서 찾은 버그 없음. 남은 설명:
1. **nll 지표의 노이즈** (가장 유력): 128토큰 nll 합은 소수 토큰이 좌우하고, tie 수준 차이만으로 ±4 % 움직임. → 토큰 단위 짝 비교 스크립트 `ppl_compare.py` (부호 검정, 짝 bootstrap 95 % CI, 차이 큰 5 % 제외 비율)
2. host에서 재현 안 되는 DSP 동작 (HMX 수치 등) → 6회차 B의 `NNTR_HTP_ATTN_F32_CHECK=1` (DSP 경로 대 NEON 경로, 같은 scale)

## 첫 토큰 분석

- prefill과 decode가 expert pool 하나(층당 16 × 30층 = 480 slot)를 공유. prefill은 층마다 128개 expert를 거의 다 읽으므로 prefill이 끝나면 pool에는 마지막 몇 층 것만 남음 → 첫 토큰 miss 220/240.
- `NNTR_MOE_WARM=<n>` (`d765641`): prefill 마지막 호출의 라우팅으로 층별 n개(마지막 토큰의 top-(k+5) 순위, 그다음 많이 라우팅된 순)를 골라, graph init·seed 동안 prefetch reader가 읽고 첫 토큰 전에 등록. pool 크기는 그대로(prefill 꼬리의 slot을 씀) → 메모리 증가 없음.
- n을 정하는 데이터: 한 번의 실행에서 `NNTR_MOE_TRACE`(prefill)와 `NNTR_HTP_ROUTE_LOG`(decode)를 받아 `expert_warm_sim.py`로 n별 토큰 1–4 hit 수 계산.
- graph init 100 ms는 프로세스 첫 decode에서 한 번. 구간 분해(`f407d99`)를 보고, 메모리를 늘리지 않는 구간만 prefill 앞으로 옮길 수 있음.

---

## 6회차 요청 내용

빌드 `f407d99`. 조건은 4회차와 같음 (C=16, greedy, EOS 끔, 128토큰, max_seq_len 2560, `NNTR_HTP_PAGE_CACHE=0`).

- A. KDIM calib 시간·prefill TPS (`6f71c9b` 확인)
- B. KDIM `NNTR_HTP_ATTN_F32_CHECK=1` (DSP 경로 대 NEON 경로)
- C. nll: 4회차 `$P/<prompt>.cpu.ids`에 forced 5칸, `ppl_compare.py`로 판정
- D. 첫 토큰: 라우팅 로그 1회 → `expert_warm_sim.py`, `NNTR_MOE_WARM=8 / 16` 측정, graph init 분해 줄

---

## 6회차 결과 (build `f407d99`, 모델 `gemma4-26b-a4b-tl2-rotated`, 기기 R5KL309ALRK)

기준 경로는 이 모델로 만든 `cpu.ids` (4회차 파일은 다른 모델).

**A. KDIM 비용 (p2k)**: calib 합 61.3 → 112.8 ms, accel_call_us 618 → 697 ms, prefill TPS 654.6 → 645.9 (−1.3 %). HVX 병합으로 133 → 113 ms만 줄었고 남은 비용이 있음.

**B. DSP 대 NEON (`F32_CHECK`, p1k)**: off와 KDIM 모두 층마다 0.04~0.12 %의 u16 원소가 1 step 이상 다름 (worst 4~32 step), KDIM이 off의 약 1.3배. KDIM 전용 DSP 버그 없음.

**C. nll (`ppl_compare.py`)**

| 비교 | p1k sum ratio [95 % CI] | verdict | p2k | verdict |
|---|---|---|---|---|
| KDIM / off | 1.084 [0.98, 1.23] | not worse | 0.983 [0.85, 1.10] | undecided |
| KDIM+SEED_Q / off (노이즈) | 1.009 [0.92, 1.12] | undecided | 1.000 [0.90, 1.12] | undecided |
| KDIM / off, host calib | 0.988 [0.89, 1.08] | not worse | 0.953 [0.86, 1.05] | undecided |

"B worse" 없음, 부호 검정 p ≥ 0.25, 중앙값 차이 0. 128토큰 nll의 CI 폭이 ±10~20 %라 2 % 게이트는 이 지표로 못 가림 (4회차의 1.115도 이 폭 안). 5회차의 층별 오차(KDIM이 절반)와 합쳐 **KDIM 통과**.

**D. 첫 토큰 (p1k)**

- `expert_warm_sim.py`: token 1의 240개 중 warm 목록에 든 수 n=4: 63, n=8: 109, n=13: 129, n=16: 140 (token 2~4는 n=16에서 94/76/79)
- `NNTR_MOE_WARM`: warm 줄 없음, misses 9회 모두 219 → hook이 Gemma 4의 층(`Lfm2MoELayer`)에 없던 버그. `1763aa6`에서 수정
- 같은 misses에서 miss_us가 70 ms와 115~130 ms로 갈림 (파일 읽기 지연의 차이로 보임) → 비교는 중앙값, 반복 수를 늘림
- graph_init 105~110 ms = params 44~50 (router 가중치 등 load 파라미터 RPC, 약 43 MB) + attn 55~62 (prefill int8 cache 해제, decode cache 등록) + rpc 1 + e2e 2
  - attn과 seed(96 ms)는 C 구조가 없애는 몫

## 7회차 (요청 예정): warm-up 재측정

`NPU/round7-prompt.md`

## 남은 일

| 순서 | 일 | 조건 |
|---|---|---|
| 1 | warm-up 효과 측정, n 결정, 기본값 여부 | 7회차 |
| 2 | C 구조 S1–S7 (decode 형식 masters, prefill RoPE·양자화 DSP, host fp16 KV 삭제, 2배 단위 재양자화, DSP compaction, SEED_Q 삭제): 첫 토큰의 seed 96 ms와 graph_init attn 57 ms를 없앰 | KDIM 통과 → 시작 |
| 3 | KDIM calib의 남은 +51 ms (p2k): worker와 병합 구간 분해부터 | 측정 |
| 4 | graph_init params 45 ms (load 파라미터 복사) | 메모리 영향 확인 후 |
| - | DSP fp16 attention 커널 NaN (3회차) | 별건 |

---

## host에서 확인한 것 (누적)

| check | 결과 |
|---|---|
| `test/htp/host/run_host_checks.sh` (각 커밋) | 통과 |
| `Q2 QPREP KD OK` | DSP 커널의 KDIM Q 접기 = host 공식 (tile 바이트), mutant 3개 |
| `ExpertWarm.*`, `ExpertLru.*` | warm 목록 규칙, warm-up의 LRU 순서, mock backend로 note→install→begin→end (20/20), mutant 5개 |
| `Q2C CALIB WORKERS OK` | 두 calibration worker를 소스에서 추출해 스칼라 참조와 lane 단위 비교. mutant 2개 잡힘 |
| `unittest_hexkl_kv_q` | 7/7 (`FixedScalesKPerDim` 포함) |
| `ATTN M1 SEED Q8 KD OK` | masters seed == fp16 재양자화 (scale 비트 동일, K 바이트는 tie만 차이) |
| `Gemma4HtpPplDecode` | tiny 모델에서 forced 실행 nll_sum == self, 바꾼 id가 읽힘. skip_prefill on/off. mutant 2개 잡힘 |
| `unittest_causallm_models` | 109 통과, 15 skip (Q4_0, `NNTR_QUANTIZE_BIN` 없음) |
