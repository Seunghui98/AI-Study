# 첫 토큰 대기 줄이기 + KV cache fp16 제거 — 구현 계획

기준 코드: nntrainer PR #4428 head `a3640d2` (2026-10-21)
분담: 구현·host 검증은 Claude, 실기기(S26U) 측정·판정은 사용자
원칙: 단계마다 env 스위치로 on/off, 기본값 off. 기기에서 같은 바이너리로 A/B 측정 → 통과하면 다음 단계에서 기본 on.

관련 분석: `kv-cache-int8-direct-analysis.md` (7절 fp16 제거, 8절 첫 토큰 숨기기)

---

## 전체 그림

```
Phase 0  측정           t1 로그로 첫 토큰 506 ms 분해            (코드 변경 없음)
Phase 1  빠른 것        growPool 비동기 / graph_init 조기 / poolSync 겹치기   (정확도 영향 없음)
Phase 2  K scale dim당  prefill attention의 K scale을 (head, dim)당으로, Q에 접기   (정확도 개선 방향)
Phase 3  seed 재배치    seed를 fp16 재양자화 → int8 masters 재배치, prefill 중 background 실행
Phase 4  fp16 제거      host fp16 cache를 chunk 1개 분량으로, snapshot/compaction을 DSP cache 기준으로
Phase 5  첫 토큰 직송   CPU 그래프 걷기 없이 첫 토큰을 패킷으로
```

Phase 1은 독립. Phase 3은 2가 전제. Phase 4는 3이 전제. Phase 5는 1·3이 전제.

---

## Phase 0. 측정 (사용자)

현재 head 그대로, p1024 / p2048, 3회씩.
```
NNTR_HTP_PROFILE=1 ./nntr_causallm <pack>
```
읽을 줄:
- `[HTP] t1: graph_init_ms=`
- `[HTP] t1: pool_sync_ms= (drop_host_fc+pin ...) kv_seed_rpc_ms= seeds=`
- `[HTP] token driver: first token pos= id= us=`
- `[TIMING]` 첫 토큰 vs 나머지

산출: 506 ms가 graph_init / seed / poolSync / 토큰 패킷 / 기타로 몇 ms씩인지. 이 값으로 Phase 1에서 graph_init 조기화를 할지 정한다.

---

## Phase 1. 빠른 것 (정확도 영향 없음)

### 1-A. growPool 비동기
- 지금: 첫 토큰 패킷 반환 직후 `growPool()`이 동기로 slot mmap + 등록 → 2번째 토큰 지연
- 변경: ARM 스레드에서 실행. 새 slot은 준비되면 pool에 추가. slot 테이블 갱신은 miss round와 lock으로 직렬화.
- 파일: `htp_compute_ops.cpp` (`growPool`, `forward`), `lfm2_moe_pool_layer.cpp`/`expert_lru.h`의 grow 콜백
- 스위치: `NNTR_HTP_GROW_ASYNC=1`
- 검증: host `expert_prefetch_check`, `moe_opts_host_check` 통과 + 기기에서 2번째 토큰 시간, 전체 decode TPS 불변, ids 동일
- 측정 지표: 토큰 2의 wall time (지금은 토큰 1 뒤 grow 시간이 붙음)

### 1-B. graph_init을 prefill 초반으로
- 지금: 첫 decode hook에서 `ensureGraphInit` (graph_init RPC + expert 테이블·load 파라미터 setParam + attention cache 등록 + e2eStart)
- 변경: prefill 첫 chunk에서 모든 MoE 층이 expert를 넘긴 시점(`moe_bound_ == moe_ops_.size()`, 층 29의 첫 MoE 호출 직후)에 graph_init + setParam + attention cache 등록까지 실행. `e2eStart`(token driver spin)만 첫 토큰 직전으로 남김.
- 주의: decode attention cache는 layer_alloc이 lazy라 메모리는 seed 때 잡힘 → 조기화해도 prefill 중 메모리 안 늘어남. graph scratch(logits 1 MiB 등)만 증가.
- 파일: `htp_compute_ops.cpp` (`ensureGraphInit` 분리, `gemm_qs4cx_moe_layer_fp32`의 bind 지점)
- 스위치: `NNTR_HTP_EARLY_GRAPH_INIT=1`
- 검증: host `graph_host_check`, `mailbox_host_check` + 기기 ids 동일
- 측정: `graph_init_ms`가 첫 토큰에서 빠지는지

### 1-C. poolSync의 ARM 일 겹치기
- `dropHostFc`, `pinExperts`, prefetch 초기화를 첫 seed RPC가 나가는 시점에 ARM 스레드로 시작, 토큰 패킷 직전 join.
- 스위치: `NNTR_HTP_POOLSYNC_ASYNC=1`
- Phase 0에서 `pool_sync_ms`가 10 ms 미만이면 생략.

Phase 1 완료 기준: 첫 토큰에서 graph_init + poolSync 몫이 빠지고 ids 동일.

---

## Phase 2. prefill K scale을 (head, dim)당으로

목적: prefill cache와 decode cache의 scale 규격을 같게 → Phase 3에서 seed가 손실 없는 재배치가 됨. 부수 효과로 prefill K 양자화가 더 촘촘해짐.

수학: score = Σ_d q_d·k_d, k_d = s_d·k8_d → Σ_d (q_d·s_d)·k8_d. K의 dim별 scale은 Q를 u16으로 만들기 전에 Q에 곱한다. HMX 쪽·colsum·zp 보정항은 변경 없음.

변경 목록

| 파일 | 변경 |
|---|---|
| `hmx/hexkl_kv_q.h/.c` | `fs_k[n_kv]` → `[n_kv][hd]`, `fs_k_inv` 추가. `quant_k_row_fixed`가 dim별 inv 벡터 사용 (V와 같은 형태). `set_fixed_scales` 인자 길이 변경 |
| `hmx/hexkl_attn_q2.c` | Q 준비(`q_f32_to_u16`, 247~292행)에 dim별 `s_d` 곱 추가 (hd/32 벡터). `alpha`(675행)에서 `fs_k[n]` 제거. u16 Q 경로(`attn_q2_step`)는 host가 접은 Q를 주도록 |
| `test/htp/nntr_hvx_attn_q.c` | DSP calibration(560~700행) `kmax[lane][n_kv]` → `[n_kv*hd]` (vmax와 같은 코드). `kv_set_fixed_scales_q` 길이 검사 |
| `test/htp/nntr_hvx.idl` | `kv_set_fixed_scales_q`의 s_k 길이 의미 변경 (sequence라 시그니처는 그대로, 검사만). 새 entry 없음 |
| `htp_mha_core.cpp` | `q2_kmax` per head → per dim (`abs_max_f16_lanes` 재사용), `q2_derive_kv`, `q2_derive_q`는 접은 Q 기준으로, `.q2` 사이드카 저장/로드 포맷 버전 올림 (K scale 길이 변경) |
| `cpu_backend/cpu_kv_q_attention.h` | CPU 미러 경로가 있으면 같은 규격으로 (없으면 생략) |

호환: 기존 `.q2` 파일(K scale head당)은 버전으로 구분해 로드 시 dim 전체에 복사(동작 동일).

검증
- host: `kv_fixed_tail_check`, `kv_dynamic_tiles_check`, `attn_qprep_native_check`, meson `unittest_hexkl_kv_q`, `unittest_cpu_kv_q_attention` 수정·통과. attn_q2 HMX 커널 자체는 host 수치 검증이 없음(stub) → **기기 검증 필수**
- 기기(사용자): `unittest_hvx_attn_q` 디바이스 gtest, p1024/p2048 greedy ids, nll(기존 int8 seed 0.2278 기준), `NNTR_HTP_ATTN_TRACE`의 full 층 CPU 대비 오차(지금 1.2~2.7%)가 같거나 감소

스위치: `NNTR_HTP_Q2_KDIM=1`. 통과 후 기본 on.

위험: Q u16 범위가 dim별로 달라짐. `q2_derive_q`를 접은 Q에 대해 잡으므로 overflow는 없지만, 작은 dim의 유효 bit가 줄 수 있음 → nll로 판정. 실패하면 Phase 3을 "K는 fp16에서, V는 masters에서"로 축소.

---

## Phase 3. seed를 masters 재배치로 + prefill 중 background 실행

### 3-A. masters → decode cache 재배치 (손실 0)
- 경로는 이미 있음: `nntr_hvx_attn_m1_seed_kvq` → `hvx_attn_m1_seed_q8` (K도 `k_rows=NULL`로 masters에서)
- 변경:
  - `hvx_attn_m1_seed_q8`: K scale을 `fs_k[n_kv*hd]`에서 복사 (Phase 2 규격)
  - rounding 통일: ATTN_M1의 decode append `q8_of`(+0.5 truncation) → `rintf`(RNE)로. seed는 복사라 rounding 무관하지만 이후 토큰 append가 prefill과 같은 규칙을 쓰게
  - `htp_mha_core.cpp` 779~786: seed_q 조건(`q_cache_synced[0] == pos`)이 안 맞아 조용히 fp16으로 떨어지는 경우를 로그로 드러냄 (`[HTP] seed: layer L fell back to f16 (synced=.. pos=..)`)
  - margin: prefill `NNTR_HTP_Q2_MARGIN` 기본 1.0 유지 → decode scale과 동일
- 스위치: 기존 `NNTR_HTP_KV_SEED_Q=1` 재사용
- 검증: host `attn_m1_host_check`의 SEED Q8 케이스를 K-from-masters로 확장. 기기: `KV_SEED_Q=1` vs 0에서 decode cache 바이트 덤프 비교(`NNTR_HTP_DUMP`)가 0 차이, p2048 512토큰 ids 동일 — **이전에 깨졌던 케이스 재현이 이 단계의 첫 일**. 깨지면 의심 지점(분석 문서 4절) 순서로 좁힘.

### 3-B. seed를 마지막 chunk의 층별 attention 직후 background로
- 조건: host가 "이 호출이 이 층의 마지막 prefill 행"임을 앎 (prefill 루프가 총 행 수를 앎). `sdpa_q2_kvcache` 호출에 `last=1` 플래그.
- DSP: `attn_q2_step*`가 `last` 플래그를 받으면 호출 끝에 seed job(masters → ATTN_M1 cache 재배치)을 worker pool background lane에 queue하고 리턴. ATTN_M1 cache는 그 시점에 `layer_alloc` (lazy).
- sliding 층: window 행만 재배치. 행 범위는 DSP가 `cache_to`와 window로 계산. floor 불필요(prefill scale이 전체 행 누적 |max|).
- prefill cache 해제: seed job 완료 후 그 층의 kv_q 해제 (마지막 chunk라 더 쓸 일 없음) → peak 감소.
- 첫 decode hook: `kv_len_[ord] == row`면 seed 생략(이미 됨). 층 29는 job 완료를 기다림.
- 파일: `hexkl_attn_q2.c`(flag), `nntr_hvx_attn_q.c`(job queue), `hvx_attn_m1_f32.c`(seed job 함수), `htp_compute_ops.cpp`(flag 전달, 완료 대기), `htp_mha_core.cpp`(last 판정)
- 스위치: `NNTR_HTP_SEED_EARLY=1`
- 검증: host `attn_m1_host_check` + `moe_layer_host_check`에 background job 동시 실행 케이스. 기기: `kv_seed_rpc_ms`→0 근처, 첫 토큰 ms, ids 동일, prefill TPS 손해(DDR 경쟁) 측정

---

## Phase 4. host fp16 cache 제거

전제: Phase 3-A가 기기에서 bit-identical 확인됨.

### 4-A. scale 성장 시 재양자화를 masters에서
- `q2_grow`로 scale이 커지면 지금은 host fp16에서 전체 재append. 변경: DSP가 masters를 새 scale로 재양자화 (int8→int8, 오차 ≤ 직접 양자화의 2배, 성장 이전 행만)
- 현재 chunk 행은 host가 아직 들고 있는 fp16(4-B)에서 정확히 재양자화
- 새 skel entry: `kv_requant_q(handle, old_scales, new_scales)`
- 성장 빈도 측정이 먼저: `NNTR_HTP_ATTN_TRACE`로 2k/6k/20k에서 층별 grow 횟수 (사용자)

### 4-B. host fp16을 chunk 1개 공유 버퍼로
- `KVCacheManager`가 층별 max_seq_len fp16 대신 2048행 버퍼 1개(층 간 공유)만 유지
- `compactKVCache`: host 행 이동 → DSP masters memmove + 타일 re-bake. 새 skel entry `kv_compact_q(handle, from, rows)`
- CPU fallback 경로는 fp16 전체가 없으면 불가 → HTP 경로에서는 fallback을 에러로 (지금도 사실상 안 씀)
- 스위치: `NNTR_HTP_KV_HOST_CHUNK=1`

### 4-C. snapshot을 masters + scale로
- 저장: `save_kvcache`가 kv_q masters(DSP→host 복사) + fs_k/fs_v를 씀. `.q2` 포맷 버전 올림
- 로드: register + masters 그대로 append (재양자화 없음), "mirror 재append" 제거
- 구 포맷(fp16 dump) 로드는 유지

검증: host `unittest_kv_cache_manager` 확장, `kv_fixed_tail_check`. 기기: 2k/6k/20k, save→use 시나리오 ids, peak 메모리(30k에서 fp16 1.2 GB 감소 확인), prefill TPS.

---

## Phase 5. 첫 토큰을 패킷으로 직송

전제: 1-B(graph 준비), 3-B(seed 준비), 1-C(pool 준비)가 prefill 끝에 완료돼 있음.
- `CausalLM::generate`의 첫 decode에서 CPU 그래프 걷기 대신 `forward` 패킷 1개. hook이 하던 RoPE 테이블 bind·conv state 등은 Gemma4에 해당 없음(RoPE 공유는 graph_init에서).
- 스위치: `NNTR_HTP_FIRST_TOKEN_DIRECT=1`
- 기대: 첫 토큰 ≈ steady 토큰(75~100 ms) + 층 29 seed

---

## 일정 (구현 기준, 기기 측정 제외)

| Phase | 구현 | host 검증 | 사용자 측정 항목 |
|---|---|---|---|
| 0 | - | - | t1 분해 (p1024, p2048) |
| 1 | 1~2일 | host checks | 첫 토큰 ms, 토큰 2 ms, ids |
| 2 | 2~3일 | host checks + meson | 디바이스 gtest, ids, nll, ATTN_TRACE 오차 |
| 3 | 2~3일 | host checks | seed_q 깨짐 재현 → 바이트 비교 → 첫 토큰 ms, prefill TPS |
| 4 | 3~5일 | host checks | grow 빈도, peak 메모리, snapshot 시나리오 |
| 5 | 1일 | - | 첫 토큰 ms |

Phase 2·3 사이, 3·4 사이에 기기 판정이 있어야 다음으로 간다.

---

## 결정이 필요한 것

1. 코드 전달 방법: Seunghui98/nntrainer 포크에 브랜치로 push vs AI-Study에 patch 파일
2. 범위: Phase 1~3까지 먼저(첫 토큰 목적) vs 4까지(메모리 목적) 한 번에
3. Phase 2 실패 시 축소안 수용 여부 (K는 fp16 seed 유지, V만 masters)
