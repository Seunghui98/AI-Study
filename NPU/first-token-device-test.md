# 첫 토큰 / KV int8 — 기기 측정 가이드

코드: `Seunghui98/nntrainer` 브랜치 `htp/first-token-kv-int8` (base `a3640d2`)

| 커밋 | 내용 | 스위치 (기본 off) |
|---|---|---|
| `ecb6eb4` | Phase 2. prefill int8 K scale을 (head, dim)당으로, 비율은 Q에 접음 | `NNTR_HTP_Q2_KDIM=1` |
| `f30bde9` | Phase 3-A. KDIM일 때 decode seed의 K도 int8 masters에서 (V는 원래 masters) | `NNTR_HTP_KV_SEED_Q=1` (기존 스위치) |

두 스위치를 다 끄면 동작은 base와 같다 (per-head 경로는 같은 inverse·같은 벡터를 씀).

---

## 1. 무엇이 바뀌었나

**Phase 2 (`NNTR_HTP_Q2_KDIM=1`)**
- prefill int8 K: scale `fs_k[head]` → `fs_kd[head][dim]` (decode q8 cache와 같은 단위)
- HMX 커널: alpha는 그대로 `fs_k = max_d fs_kd`, Q를 u16으로 만들 때 `fs_kd / fs_k`를 곱함 (`hexkl_attn_q2.c` stage_qprep)
- calibration: host(`q2_grow`, `q2_derive_q`)와 DSP(`attn_q2_step_fqc`) 모두 K는 dim별 max, Q 범위는 접은 Q 기준
- f32 Q 경로 전용: `NNTR_HTP_ATTN_F32=0`, `NNTR_HTP_ATTN_F32_CHECK`가 있으면 자동 off. f32 호출이 실패한 층은 u16 경로 대신 fp16 경로로 감 (`mha_core: quantized KV cache attention (K per dim needs the f32 Q path) failed` 로그)
- 덤으로 고친 것: prompt가 8행보다 짧을 때 DSP calibration이 돌지 않은 lane 버퍼까지 합치던 문제

**Phase 3-A (`NNTR_HTP_Q2_KDIM=1` + `NNTR_HTP_KV_SEED_Q=1`)**
- 첫 decode seed: K·V 모두 DSP의 int8 masters를 그대로 decode cache로 복사. host fp16 K → f32 변환 없음
- KDIM 없이 `KV_SEED_Q=1`만 켜면 예전처럼 K는 fp16에서
- seed가 masters를 못 쓰는 층은 이제 이유를 로그로 남김:
  ```
  mha_core: <layer> seeds decode from fp16, not the int8 cache (cache_base=.. handle=.. synced=.. pos=.. scales_set=..)
  ```
  DSP가 거절하면 기존 로그: `[HTP] attn_m1_seed_kvq: layer N not seeded on the DSP (...)`

**하지 않은 것 (이유)**
- 1-A growPool 비동기: `growPool`은 `addCapacity` + session_info RPC뿐, slot은 miss 때 lazy로 잡힘 → 할 일 없음
- 1-B graph_init 조기화, 1-C poolSync 겹치기: `ensureGraphInit`이 decode FC arena와 attention cache 등록까지 하므로 prefill 중에 하면 prefill int8 cache와 함께 PD(3840 MiB)에 올라감. 이득은 아래 0번의 `graph_init_ms`, `pool_sync_ms`를 보고 정함
- 3-B seed를 prefill 마지막 chunk 중 background로: decode attention ctx가 graph_init 때 생기므로 1-B가 전제. 3-A 후 `kv_seed_rpc_ms`가 작으면 불필요
- Phase 4 (host fp16 제거): 계획상 3-A가 기기에서 맞다는 확인 + scale 성장 빈도 측정이 먼저. 지금 host fp16은 int8 mirror 재append(scale 성장, compaction 뒤)의 원본이라, 없애려면 DSP 쪽 재양자화·compaction·snapshot이 필요
- rounding 통일 (`q8_of` → RNE): masters(RNE)와 decode append(half-away)는 정확히 .5인 값에서만 다르고 둘 다 최근접. host check가 차이가 tie에서만 나는 것을 확인함

---

## 2. host에서 확인한 것

| check | 결과 |
|---|---|
| `test/htp/host/run_host_checks.sh` (P2, P3 각각) | 통과 |
| `unittest_hexkl_kv_q` (meson 대신 gtest 직접 빌드) | 7/7, 새 `FixedScalesKPerDim`: append/dump, `fs_k·Σ(q·r·k8) == Σ(q·s_kd·k8)`, per-head로 되돌리기. mutant 3개 잡힘 |
| `attn_m1_host_check` 새 `ATTN M1 SEED Q8 KD OK` | masters(dim별 amax/127, RNE) seed == `kv_append` scale bit 단위 동일, K 바이트는 tie에서만 차이. head scale mutant 잡힘 |
| 바뀐 DSP 소스 `__hexagon__` 타입 체크 (hvx_emu + IDL 생성 헤더) | 경고·에러 없음 |

host에서 못 한 것: HMX 커널 수치, `kv_fixed_tail_check`(SDK libnative 필요), 실제 nll·ids.

---

## 3. 기기 측정 순서

공통: p1024, p2048 각 3회. 같은 prompt, greedy.
```
NNTR_HTP_PROFILE=1 NNTR_HTP_TOKEN_TRACE=<out>.json ./nntr_causallm <pack>
```

### 0) 기준 (스위치 모두 off) — Phase 0
읽을 줄:
- `[HTP] t1: graph_init_ms=`
- `[HTP] t1: pool_sync_ms= ... kv_seed_rpc_ms= seeds=`
- `mha_core: t1 seed pos=.. rows_convert_ms_sum=.. seed_ms_sum=..`
- `[HTP] token driver: first token pos= id= us=`

→ 첫 토큰이 graph_init / seed / poolSync / 패킷 / 기타로 몇 ms씩인지. 1-B·1-C 진행 여부를 이걸로 정함.

### 1) `NNTR_HTP_Q2_KDIM=1` — Phase 2 판정
- device gtest: `unittest_hvx_attn_q` (`test/htp/run_u8i4_layer_on_device.sh`로 빌드) 통과
- SDK가 있으면: `python3 test/htp/host/run_kv_fixed_tail_check.py ...` 통과
- ids: 0)과 같거나, 다르면 어디서 갈리는지 (`NNTR_HTP_TOKEN_TRACE`)
- nll: 기존 측정 방식으로 0)과 비교 (기존 int8 seed 0.2278 기준). 같거나 낮으면 통과
- `NNTR_HTP_ATTN_TRACE=1`: `q2 clip ... k_max/127=` 와 full 층 CPU 대비 오차(지금 1.2~2.7 %)가 같거나 감소
- prefill TPS 변화 (Q 준비에 hd/32 벡터 곱 추가, calibration lane 버퍼 증가)

### 2) `NNTR_HTP_Q2_KDIM=1 NNTR_HTP_KV_SEED_Q=1` — Phase 3-A 판정
- 위 fallback 로그가 몇 층에서 나오는지, 나오면 그 값 (sliding 층은 window 때문에 `cache_base>0`이면 fp16으로 감 — 예전 p2048 깨짐의 첫 의심 지점)
- `kv_seed_rpc_ms`, `rows_convert_ms_sum` (K 변환이 빠져야 함), 첫 토큰 ms
- ids: 1)과 비교, p2048은 512토큰까지. tie 차이만 있으므로 거의 같아야 함
- 갈리면: `NNTR_HTP_DUMP=<dir>`로 첫 decode 토큰의 층별 attention 출력을 1)과 비교해서 처음 다른 층을 찾음

### 3) Phase 4 사전 측정
- `NNTR_HTP_Q2_KDIM=1 NNTR_HTP_ATTN_TRACE=1`, 2k / 6k / 20k: `q2 scales grew at rows=` 줄 수 (층별). 거의 없으면 4-A(DSP 재양자화)는 단순 경로로 충분
- 각 길이의 peak 메모리 (prefill 중)

---

## 4. 결과에 따른 다음 단계

| 결과 | 다음 |
|---|---|
| 0)에서 graph_init_ms가 큼 | 1-B 구현 (스위치로) |
| 1) nll 같거나 개선 | KDIM 기본 on 후보, 2)로 |
| 1) nll 악화 | 축소안: K는 fp16 seed 유지, V만 masters (지금 `KV_SEED_Q=1`만 켠 상태와 같음) |
| 2) ids 같음 + seed 빨라짐 | Phase 4 설계 확정 (3)의 성장 빈도 반영), 3-B는 seed 시간이 남으면 |
| 2) fallback 로그가 sliding 층에서 | windowed cache를 masters에서 seed하는 경로 추가 (window 행만 재배치) |
