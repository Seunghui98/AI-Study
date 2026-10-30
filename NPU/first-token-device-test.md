# 첫 토큰 / KV int8 — 진행 상태와 기기 측정 기록

코드: `Seunghui98/nntrainer` 브랜치 `htp/first-token-kv-int8` (base `a3640d2`, head `ec09f64`)

| 커밋 | 내용 | 스위치 (기본 off) |
|---|---|---|
| `ecb6eb4` | prefill int8 K scale을 (head, dim)당으로, 비율은 Q에 접음 | `NNTR_HTP_Q2_KDIM=1` |
| `f30bde9` | KDIM일 때 decode seed의 K도 int8 masters에서. seed fallback 로그 | `NNTR_HTP_KV_SEED_Q=1` (기존) |
| `568fa2e` | host NEON u16 Q 경로도 같은 방식으로 접음 | - |
| `d1b6873` | 스위치 off의 DSP calibration을 a3640d2와 같게 되돌림 (`ecb6eb4`의 prefill 회귀 제거). KDIM 경로는 레지스터 누적 | - |
| `ec09f64` | `NNTR_PPL_DECODE` 다시 추가 (decode nll 게이트, skip_prefill 지원) | `NNTR_PPL_DECODE=<ids 파일>` |

스위치를 다 끄면 base와 같다 (2회차에서 ids 비트 동일, prefill 노이즈 범위 안).

---

## 목표와 계획 (C 구조)

prefill과 decode가 decode 형식의 int8 KV 한 벌만 쓴다. host fp16 KV 없음, 첫 decode의 seed 변환 없음.

1. 회귀 제거 — 완료 (`d1b6873`, 2회차 통과)
2. KDIM 정확도 게이트 — 진행 중 (2회차 게이트는 기준이 편향돼 판정 보류, 아래)
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

## 3회차 (요청 중): 편향 없는 nll 게이트

- 경로 작성: prefill attention을 fp16으로 (pack의 `attention_kv_dtype`를 `""`로 바꾼 symlink pack), `NNTR_PPL_DECODE` self
- 채점: 원래 pack의 off / KDIM / KDIM+SEED_Q가 같은 파일을 forced
- off와 KDIM은 decode seed가 같고 prefill K scale만 다름 → 같은 경로 위에서 nll이 낮은 쪽이 fp16 prefill에 가까움
- 노이즈 = KDIM+SEED_Q / KDIM (tie 수준 차이)
- 판정: KDIM / off가 노이즈 범위 안이거나 1 미만이면 C 진행, 노이즈보다 분명히 크면 C 중단
- 같이: 2회차 로그의 `[HTP] t1: graph_init_ms=`, `pool_sync_ms=`, `t1 seed`, `token driver: first token` 줄 (첫 토큰 분해)

---

## host에서 확인한 것 (누적)

| check | 결과 |
|---|---|
| `test/htp/host/run_host_checks.sh` (각 커밋) | 통과 |
| `Q2C CALIB WORKERS OK` | 두 calibration worker를 소스에서 추출해 스칼라 참조와 lane 단위 비교. mutant 2개 잡힘 |
| `unittest_hexkl_kv_q` | 7/7 (`FixedScalesKPerDim` 포함) |
| `ATTN M1 SEED Q8 KD OK` | masters seed == fp16 재양자화 (scale 비트 동일, K 바이트는 tie만 차이) |
| `Gemma4HtpPplDecode` | tiny 모델에서 forced 실행 nll_sum == self, 바꾼 id가 읽힘. skip_prefill on/off. mutant 2개 잡힘 |
| `unittest_causallm_models` | 109 통과, 15 skip (Q4_0, `NNTR_QUANTIZE_BIN` 없음) |
