# 첫 토큰 / KV int8 — 기기 실행 가이드

코드: `Seunghui98/nntrainer` 브랜치 `htp/first-token-kv-int8` (base `a3640d2`)

| 커밋 | 내용 | 스위치 (기본 off) |
|---|---|---|
| `ecb6eb4` | prefill int8 K scale을 (head, dim)당으로, 비율은 Q에 접음 (DSP f32 Q 경로) | `NNTR_HTP_Q2_KDIM=1` |
| `f30bde9` | KDIM일 때 decode seed의 K도 int8 masters에서 (V는 원래 masters). seed fallback 로그 | `NNTR_HTP_KV_SEED_Q=1` (기존) |
| `568fa2e` | host NEON u16 Q 경로도 같은 방식으로 접음 → `NNTR_HTP_ATTN_F32_CHECK`로 DSP 접기 검증 가능 | - |

스위치를 다 끄면 base와 같은 코드 경로다.

---

## 1. 빌드

`docs/gemma4-htp-user-guide.md` 4.2절과 같다.

```bash
git fetch https://github.com/Seunghui98/nntrainer htp/first-token-kv-int8
git checkout FETCH_HEAD

export ANDROID_NDK=...            # r30
export HEXAGON_SDK_ROOT=.../Hexagon_SDK/6.4.0.1
source "$HEXAGON_SDK_ROOT/setup_sdk_env.source"
export HEXKL_ROOT=.../hexkl-1.0-beta.2/hexkl_addon
export HEXKL_SDK_VER=6.4.0.2
export HEX_ARCH=v81

bash Applications/CausalLM/build_android.sh --htp --app
```

skel(`test/htp/build/libnntr_hvx_skel.so`)과 ARM 쪽(`libnntrainer.so`)이 같은 커밋에서 나와야 한다. IDL은 안 바뀌었지만 skel의 calibration 길이 검사가 바뀌었다.

---

## 2. 실행

스크립트: `NPU/run_first_token_tests.sh` (이 저장소). nntrainer 체크아웃 루트에서 실행.

```bash
cp ~/AI-Study/NPU/run_first_token_tests.sh .
bash run_first_token_tests.sh stage              # 바이너리 push (D=/data/local/tmp/htp)
bash run_first_token_tests.sh run p1024.txt      # 아래 6개 케이스 x 3회
bash run_first_token_tests.sh run p2048.txt
bash run_first_token_tests.sh summary            # 숫자만 모아 보기
```

- `p1024.txt`, `p2048.txt`: 지난 측정에 쓴 prompt를 파일로. 첫 인자 그대로 CLI의 prompt가 됨 (chat template 적용)
- pack 경로가 다르면 `PACK=/sdcard/.../pack`, 반복 횟수는 `REPS=1`
- 생성 토큰 수는 pack의 `nntr_config.json` `num_to_generate` (기본 127). p2048 512토큰 확인은 이 값을 512로 바꾼 pack으로
- 로그는 `ft_logs/<prompt>_<case>_r<n>.log`, 토큰 id는 `ft_logs/trace_*.json`

| 케이스 | 환경 | 보는 것 |
|---|---|---|
| `base` | 없음 | 기준 (Phase 0): 첫 토큰 분해, ids |
| `kdim` | `NNTR_HTP_Q2_KDIM=1` | prefill TPS, ids, nll |
| `kdim_seedq` | `+ NNTR_HTP_KV_SEED_Q=1` | seed 시간, fallback 로그, ids |
| `seedq_only` | `NNTR_HTP_KV_SEED_Q=1` | 축소안 (K는 fp16, V만 masters) 비교용 |
| `kdim_f32check` | `KDIM=1 + ATTN_F32_CHECK=1 + ATTN_TRACE=1` | DSP f32 Q 접기 == host NEON 접기 (느림, 시간 무시) |
| `kdim_hostcalib` | `KDIM=1 + ATTN_DSP_CALIB=0` | DSP calibration == host calibration |

---

## 3. 판정

### 3.1 base — 첫 토큰이 어디에 쓰이나
summary에서:
- `[HTP] t1: graph_init_ms=`
- `[HTP] t1: pool_sync_ms= ... kv_seed_rpc_ms= seeds=`
- `mha_core: t1 seed pos=.. rows_convert_ms_sum=.. seed_ms_sum=..`
- `[HTP] token driver: first token pos= id= us=`

→ graph_init / seed / pool_sync / 패킷 몫. graph_init_ms가 크면 1-B(graph_init 조기화) 진행.

### 3.2 kdim — Phase 2
| 항목 | 통과 기준 |
|---|---|
| `kdim_f32check` 로그의 `mha_core f32 edge check: ... mismatches=` | 0 (DSP 접기와 host 접기가 같은 연산). 0이 아니면 `>= one u16 step` 수와 층 이름 보고 |
| `kdim_hostcalib` vs `kdim` ids | same (calibration이 DSP든 host든 같은 scale) |
| `kdim` vs `base` ids | 같으면 좋고, 다르면 몇 번째 토큰에서 갈리는지 |
| nll | 기존 방식으로 base와 비교. 같거나 낮으면 통과 (기존 int8 seed 0.2278) |
| prefill TPS | base 대비 하락이 1~2 % 안 |
| `quantized KV cache ... failed` 로그 | 없어야 함 |

### 3.3 kdim_seedq — Phase 3-A
| 항목 | 기대 |
|---|---|
| `seeds decode from fp16, not the int8 cache` 줄 | 몇 층, 어떤 값인지 기록. sliding 층에서 `cache_base>0`이면 그 층은 fp16 seed (예전 p2048 깨짐 첫 의심 지점) |
| `[HTP] attn_m1_seed_kvq: ... not seeded` | 없어야 함 |
| `t1: ... seeds= (dsp N ...)` | N = masters로 seed된 층 수 |
| `rows_convert_ms_sum` | kdim보다 크게 줄어야 함 (K 변환 없음) |
| 첫 토큰 us | kdim보다 줄어야 함 |
| ids vs `kdim` | 거의 같아야 함 (차이는 .5 tie rounding뿐). 다르면 아래 |

ids가 다르면: `kdim`과 `kdim_seedq`를 `NNTR_HTP_DUMP=<dir>`를 더해 한 번씩 돌려 첫 decode 토큰의 층별 attention 출력 덤프를 비교, 처음 다른 층을 찾는다.

### 3.4 Phase 4 사전 측정 (선택)
`NNTR_HTP_Q2_KDIM=1 NNTR_HTP_ATTN_TRACE=1`로 2k / 6k / 20k prompt:
- `q2 scales grew at rows=` 줄 수 (층별) → 4-A(DSP 재양자화) 설계
- `peak memory:` → host fp16 제거 이득 추정

---

## 4. 결과 보내줄 것

`bash run_first_token_tests.sh summary > summary.txt` 결과와 nll 수치. 그걸로:

| 결과 | 다음 |
|---|---|
| graph_init_ms 큼 | 1-B 구현 |
| f32check mismatches 0, nll 같거나 개선 | KDIM 기본 on 후보 |
| f32check mismatches > 0 | stage_qprep 접기 수정 (층 이름·수치로 좁힘) |
| nll 악화 | 축소안 (`seedq_only`와 같은 상태) 유지 |
| seedq ids 같음, 첫 토큰 단축 | Phase 4 설계 (3.4 반영). seed 시간이 아직 크면 3-B |
| fallback이 sliding 층에서 | windowed cache를 masters에서 seed하는 경로 추가 |

---

## 5. host에서 이미 확인한 것

| check | 결과 |
|---|---|
| `test/htp/host/run_host_checks.sh` (세 커밋 각각) | 통과 |
| `unittest_hexkl_kv_q` | 7/7. 새 `FixedScalesKPerDim`, mutant 3개 잡힘 |
| `ATTN M1 SEED Q8 KD OK` | masters seed == fp16 재양자화: scale bit 동일, K 바이트는 .5 tie에서만 차이. mutant 잡힘 |
| `abs_max_check` (x86 + aarch64 qemu) | dim별 inv 양자화 NEON == scalar. mutant 잡힘 |
| 바뀐 DSP 소스 `__hexagon__` 타입 체크 | 에러·경고 없음 |

host에서 못 한 것: HMX 커널 수치, nll/ids. `unittest_hvx_attn_q` 디바이스 gtest는 이 트리의 `test/jni/Android.mk`에 빌드 타겟이 없고, 있어도 f32 Q 경로·DSP calibration을 안 다룬다 → 그 역할을 `kdim_f32check`, `kdim_hostcalib`가 한다.

## 6. 이번에 안 한 것

- 1-A growPool 비동기: slot은 miss 때 lazy로 잡혀서 할 일이 없음
- 1-B graph_init 조기화 / 1-C poolSync 겹치기: graph_init이 decode FC arena·attention cache를 올려서 prefill 중엔 3840 MiB PD에 같이 올라감. 3.1 숫자로 결정
- 3-B background seed: 1-B 전제. 3.3에서 seed가 이미 작으면 불필요
- Phase 4 host fp16 제거: 3.3 확인과 3.4 측정이 먼저
- rounding 통일: 차이가 tie뿐이고 둘 다 최근접 (host check로 확인)
