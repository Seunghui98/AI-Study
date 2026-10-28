# KV cache: fp16 원본 없이 int8로 바로 쓰기 — 코드 검토

대상: nntrainer PR #4428 HTP 경로 (Gemma4 26B-A4B)
목적: prefill 동안 host에 두는 fp16 KV cache를 없애고, int8 cache를 변환 없이(또는 손실 없는 재배치만으로) decode까지 쓰는 방법. 정확도 drop 없이.

결론 먼저
- "변환 없이"는 지금 구조로는 불가. 두 int8 cache의 scale 규격이 다르고(K), scale이 prefill 끝까지 바뀔 수 있어서다.
- 규격을 맞추면 "fp16 없이, int8 → int8 손실 없는 재배치"까지 간다. 단, scale이 중간에 커지는 경우에만 bounded error가 남는다.
- bit-exact를 끝까지 지키려면 scale을 미리 아는 방법(static KV scale)뿐이고, 그건 지금 수치와 달라지므로 별도 평가가 필요하다.

---

## 1. 지금 구조 (코드 기준)

prefill 동안 층마다 cache가 세 벌 있다.

| cache | 위치 | 형식 | scale | 원소당 바이트 | 코드 |
|---|---|---|---|---|---|
| host 원본 | ARM rpcmem | fp16 | 없음 | 2 B | `htp_mha_core.cpp:278-298` (cache_key/value FP16 or UINT16) |
| prefill cache (`hexkl_kv_q`, fixed-scale) | DSP heap | int8 offset-binary (q+128) | K: **head당 1개** (`fs_k[n_kv]`), V: (head, dim)당 (`fs_v[n_kv][hd]`) | masters 1 B + HMX 타일 1 B = 2 B | `hexkl_kv_q.h:30-33, 93-108`, `hexkl_kv_q.c:196-232` |
| decode cache (`ATTN_M1` q8) | DSP heap | int8 signed | K, V 모두 (head, dim)당 (`sk/sv[layer][n_kv*hd]`) | 1 B | `hvx_attn_m1_f32.c:411-444` |

흐름
```
QKV proj (f32)
 ├─ host fp16 cache에 행 추가 (NEON f32→fp16)              htp_mha_core.cpp:1843~
 └─ prefill int8 cache에 행 추가 (fixed scale로 양자화)       sdpa_q2_kvcache → kv_q append
첫 decode 토큰
 └─ seed : host fp16 행 → |max|로 decode scale → int8 → ATTN_M1 cache   htp_mha_core.cpp:796-826
           이후 host fp16 해제 (KV_DROP_CPU), prefill int8 cache 해제 (FREE_KVQ)
```

30k context 기준 prefill 중 메모리: fp16 약 1.2 GB + prefill int8(masters+타일) 약 1.2 GB.

## 2. "변환 없이"를 막는 세 가지

### 2.1 K scale 단위가 다르다
- prefill: K는 head당 scale 하나. 이유는 HMX 커널 구조 — K scale이 softmax의 residual factor에 Q scale과 함께 들어간다 (`hexkl_attn_q2.c:675`: `alpha = s_q * fs_k[n] * LOG2E / sqrt(hd)`), 그리고 Q scale이 head당이라 K도 head당.
- decode: (head, dim)당.
- 이미 해 본 실험 `NNTR_HTP_KV_SEED_Q=1` (b35c5a6, `hvx_attn_m1_seed_q8`): V는 prefill int8을 XOR 0x80만 해서 그대로 가져오고(`hvx_attn_m1_f32.c:738-740`), K는 head당 scale 그대로 쓰면 nll +9.4%라 fp16에서 다시 양자화. 결과: nll 0.2344(fp16 seed 0.2331, 기본 int8 seed 0.2278)로 비슷했지만 p2048 512토큰 출력이 깨짐. 원인 미기록.

### 2.2 scale이 prefill 끝까지 바뀐다
- prefill cache의 fixed scale은 첫 step 행으로 calibration하고, 뒤 chunk에 더 큰 값이 오면 scale을 키우고 **host fp16에서 전체를 다시 양자화**한다 (`htp_mha_core.cpp:1204-1230 q2_grow`, `1574-1583 synced=0 재append`). 2k 문서에서 V가 2배까지 넘은 기록(eb91939).
- decode cache scale은 seed 때 전체 행 |max|로 한 번 정하고 고정 (`q8_set_scales_rows`, `scaled[layer]`).
- 즉 fp16 원본이 있어야 "scale이 커졌을 때 손실 없이 다시 양자화"할 수 있다. 이게 fp16을 못 없애는 진짜 이유다.

### 2.3 저장 형식이 다르다
- prefill masters: offset-binary(q+128), row-major `[n_kv][rows][hd]`.
- decode: signed int8, K는 64위치 단위 전치 타일 (`kt[d*TILE + pos%TILE]`), V는 row-major.
- 이건 손실 없는 정수 재배치(XOR 0x80 + 전치)로 해결된다. `run_seed_q8`이 이미 그 코드다.

## 3. 제안: 단계별

### Step 1. prefill K scale을 (head, dim)당으로 — Q에 접기 (정확도 영향 없음, 오히려 개선)

수학: score = Σ_d q_d k_d, k_d = s_d · k8_d (dim별 scale) → Σ_d (q_d s_d) k8_d.
즉 K의 dim별 scale은 **Q를 u16으로 양자화하기 전에 Q에 곱해 두면** HMX 쪽은 바뀌지 않는다.

- Q 준비: `hexkl_attn_q2.c:247-292 q_f32_to_u16` — 지금 `x * inv(head) + zp`. 여기에 dim별 벡터 `s_d`를 한 번 더 곱한다 (hd/32개 벡터, 비용 무시 가능).
- zp 보정항 `zp * colsum_k`: colsum은 Σ_d k8_d 그대로 int. 바뀌지 않음.
- residual factor: `fs_k[n]` → 1. `alpha = s_q' * LOG2E / sqrt(hd)`.
- calibration: DSP 쪽 `nntr_hvx_attn_q.c:560-700`에서 `kmax[lane][n_kv]`를 `vmax`처럼 `[n_kv*hd]`로. host 쪽 `q2_grow`의 `abs_max_f16` → `abs_max_f16_lanes` (V와 같은 함수).
- Q u16 범위: q'_d = q_d·s_d라 dim마다 크기가 달라지지만 u16은 65,536 단계라 여유가 크다. 단, Q encoding(min/max per head)은 접은 뒤 값으로 잡아야 한다 (`q2_derive_q`).
- 정확도: K 양자화가 head당 → dim당으로 **더 촘촘해진다**. decode가 이미 dim당이므로 prefill/decode 수치 차이도 줄어든다.

### Step 2. seed를 fp16 재양자화 → int8 masters 재배치로 (손실 0)

Step 1 뒤에는 prefill cache와 decode cache의 scale 단위가 K, V 모두 같다. seed는:
```
decode sk/sv  = prefill fs_k(dim별)/fs_v        (복사)
decode K/V    = masters XOR 0x80, K는 64위치 전치   (run_seed_q8의 else 분기)
```
- 바이트가 같아지는 조건: scale이 같고 rounding이 같아야 한다.
  - scale: prefill fixed = 전체 행 |max| × margin(기본 1.0) / 127, decode = |max| / 127 (+ window floor). margin 1.0이면 동일. window floor는 Step 4에서 다룸.
  - rounding: prefill `rintf` (RNE, `hexkl_kv_q.c:104,120`), decode `q8_of` (+0.5 truncation = half away, `hvx_attn_m1_f32.c:320-324`). 정확히 .5인 값만 다르다. 하나로 통일하면 bit-identical.
- `nntr_hvx_attn_m1_seed_kvq` (`nntr_hvx_attn_m1.c:162-190`)가 이미 masters → ATTN_M1 경로이고 끝에 prefill cache를 해제한다. K도 masters에서 가져오도록 `k_rows=NULL`로 부르면 된다.
- 효과: seed RPC(p1024 약 90 ms)가 fp16 전송 + |max| pass + 양자화에서 int8 복사 + 전치로 줄어든다. host fp16 → DSP 전송 바이트도 0.

### Step 3. scale 성장(growth) 정책 — 여기가 정확도의 핵심

fp16을 없애면 뒤 chunk에서 scale이 커질 때 앞 행을 **int8에서 다시 양자화**해야 한다.

오차 bound: 원래 양자화 오차 |e1| ≤ s_old/2, 재양자화 오차 ≤ s_new/2 → 합 < s_new. 즉 **성장 이전 행만, 최대 직접 양자화의 2배**. 성장이 없으면 0.

선택지

| 정책 | 정확도 | 메모리 | 비고 |
|---|---|---|---|
| A. int8 → int8 재양자화 | 성장 발생 시 앞 행에 bounded error | fp16 0 | 가장 단순. 성장 빈도가 관건 (2k 문서에서 V 2배 사례 있음) |
| B. A + 첫 chunk calibration에 margin (예 1.5×) | 성장이 드물어짐, 대신 모든 행이 log2(1.5)≈0.6 bit 손해 | fp16 0 | `NNTR_HTP_Q2_MARGIN` 이미 있음. 측정으로 결정 |
| C. A + 현재 chunk만 fp16 유지 | 성장 시 현재 chunk는 정확, 앞 행만 bounded | fp16 16 MiB (2048행 × 2048 × 2 B × K,V, 층 간 공유 버퍼 1개) | 1.2 GB → 16 MiB. 추천 |
| D. static KV scale (`kv_quant_file`) | 성장 없음, bit-exact 재현. 단 지금 수치와 다름 (K/V head당, 층당 고정) | fp16 0 | Gemma4 CPU/GPU 경로는 이미 읽음 (`gemma4_causallm.cpp:294-350`), HTP는 미사용. checkpoint가 그 clipping으로 학습됐다면 오히려 더 충실. V가 dim당 → head당으로 거칠어지는 건 평가 필요 |

C가 "정확도 drop 없이"에 가장 가깝다: 성장이 안 일어나면 지금과 bit 동일, 일어나면 앞 행에만 ≤ 2배 오차. 성장 자체는 `NNTR_HTP_ATTN_TRACE`가 이미 기록하므로 빈도를 측정할 수 있다.

### Step 4. sliding 층 window / compaction / floor

- 지금 compaction은 host fp16 행을 옮기고(`KVCacheManager::compact`) mirror를 재append한다. masters를 원본으로 삼으면 DSP에서 masters memmove + 타일 재bake. 새 skel entry 하나.
- window floor(0428a51): 버린 행의 |max|를 decode scale에 반영. Step 2에서 scale은 prefill fixed scale을 그대로 쓰므로, prefill fixed scale 자체가 이미 전체 행 |max|(성장 누적)를 반영한다 → floor가 자연히 포함된다. 단 compaction 뒤 "현재 window 기준으로 다시 calibration"하는 길은 막힌다 (지금도 안 함).

### Step 5. snapshot
- 저장: masters + fs_k/fs_v (`.q2`에 scale은 이미 저장) → fp16 dump 불필요.
- 로드: register + masters 그대로 → "mirror 재append" 불필요, load 시간 감소.
- 부수 효과: 문서에 적힌 "snapshot/chunk 실행과 단일 prefill의 출력 차이"(calibration이 보이는 행에 의존)가 줄어든다. 로드된 상태가 저장 시점과 바이트 단위로 같아지기 때문.

## 4. 선행 과제: KV_SEED_Q 출력 깨짐 재현

Step 2가 이 경로 위에 서므로 먼저 원인을 잡아야 한다. 코드만 읽어서 보이는 의심 지점:
1. `htp_mha_core.cpp:779-780` 조건 `q_cache_synced[0] == pos`: chunk prefill 뒤 synced와 pos가 어긋나면 fp16 경로로 조용히 떨어진다. 깨진 실행이 어느 경로였는지 로그(`attn_m1_seed_kvq`)로 확인.
2. `HVX_ATTN_M1_LAZY` (seed_q일 때 ATTN_M1 lazy 등록): 등록 순서와 `layer_alloc` 타이밍.
3. `q2_grow`로 scale이 커진 뒤 masters 재양자화(synced=0 재append)가 **같은 step 안에서** 끝났는지 — seed가 재append 전 masters를 읽으면 구 scale 바이트에 신 scale이 붙는다.
4. 실험이 V만 int8에서 가져왔고 K는 fp16 + decode |max| scale이었다. K/V scale 출처가 다른 상태였으므로 Step 1 뒤 재실험이 필요하다.
host check(`ATTN M1 SEED Q8 OK`)는 통과하므로 바이트 변환 자체는 맞다. 기기에서 p2048 512토큰으로 재현하고 1~3을 하나씩 끈다.

## 5. 검증 계획

| 항목 | 방법 | 기준 |
|---|---|---|
| Step 1 정확도 | p1024/p2048 greedy 출력 id, nll (기존 int8 seed 0.2278) | nll 같거나 개선. 출력 id는 near-tie에서 바뀔 수 있음 → nll로 판단 |
| Step 2 bit-identical | rounding 통일 후 decode cache 바이트 비교 (seed_fp16 vs seed_masters) | 성장 없는 prompt에서 0 byte 차이 |
| Step 3 성장 빈도 | `NNTR_HTP_ATTN_TRACE`로 2k/6k/20k prompt에서 층별 grow 횟수와 비율 | 정책 A/B/C 결정 근거 |
| 메모리 | 30k prefill 중 peak (host anon + DSP) | fp16 1.2 GB 감소 확인. peak가 fp16이 아니라 staging/read-ahead일 수도 있으니 같이 측정 |
| 첫 토큰 | `kv_seed_rpc_ms` | 93 ms → 감소 |
| long context 회귀 | 22k+6k, 19k 단일, KV snapshot save/use | 기존 ids md5와 비교 |

## 6. 예상 효과 / 비용

- 메모리: prefill 중 fp16 1.2 GB(30k) → 정책 C면 16 MiB. 2k에서는 약 250 MB → 16 MiB.
- 첫 토큰: seed가 int8 재배치만 남아 수십 ms 단위 (미측정).
- prefill 정확도: K dim별 scale로 개선.
- 비용: attn_q2 Q 준비/calibration 수정(중), seed 경로 정리(소), DSP compaction(중), snapshot 포맷 변경(중). 커널 2개와 skel entry 1~2개.

## 7. 하지 않는 게 나은 것

- decode 커널(ATTN_M1)이 prefill 타일을 직접 읽게 하기: seed 비용만 추가로 아끼는데 커널을 다시 짜야 한다. Step 2로 충분.
- decode를 HMX row-blocked 커널로 돌리기 (`hexkl_kv_q.h` 주석에 가능하다고 적힘): 행 1개를 HMX에 넣는 건 ATTN_M1(6.7 ms/token)보다 느릴 가능성이 높다.
