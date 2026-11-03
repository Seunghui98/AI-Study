# 18회차 측정 요청: 새 #4428 위로 옮긴 브랜치 확인 (머지 전)

#4428이 리팩터링되어(파일 이동·분리, 환경 변수 → `nntr_config.json` 설정) 우리 작업 34커밋을 그 위로 옮겼다.
새 브랜치 `htp/kv-int8-4428` (기존 `htp/first-token-kv-int8`는 그대로 둠). 옮기면서 쓰지 않는 분기를 지움:
- trim fallback 전체 (`kv_trim_q` IDL·DSP·backend, 첫 토큰에서 trim된 masters로 seed): prefill seed가 항상 대신함
- kv_seed_q 모드용 KDIM 분기 (adopt가 켜지면 도달 불가)
- `NNTR_HTP_KV_INT8_ONLY` (17회차 뒤 삭제된 것 그대로)

**중요: 새 #4428에서는 `NNTR_HTP_*` / `NNTR_MOE_*` 환경 변수가 효과가 없다.** HTP 옵션은 모델 pack의 `nntr_config.json`
안 `"htp"` / `"moe"` 객체로 준다 (docs/gemma4-htp-user-guide.md "Runtime tuning"). CausalLM 쪽 환경 변수
(`NNTR_PPL_DECODE`, `NNTR_PREFILL_CHUNK`, `NNTR_GENERATED_IDS`, `NNTR_NUM_THREADS`)는 그대로 쓴다.

| 예전 환경 변수 | 새 설정 (`nntr_config.json`) |
|---|---|
| `NNTR_HTP_PAGE_CACHE=0` | `"htp": {"page_cache": 0}` |
| `NNTR_HTP_PROFILE=1` | `"htp": {"profile": 1}` |
| `NNTR_HTP_ATTN_TRACE=1` | `"htp": {"attn_trace": true}` |
| `NNTR_HTP_KV_ADOPT=0` | `"htp": {"kv_adopt": false}` (새 필드) |
| `NNTR_MOE_WARM=0` | `"moe": {"warm_experts": 0}` (새 필드, 기본 16) |

목적: 옮긴 빌드가 17회차(`d283f77b6`)와 같게 도는지. 기대:
- 한 블록 prompt의 step 줄이 17회차 int8only(= 15회차 int8only)와 같음. 다르면 #4428 쪽 변경 때문인지 off 셀로 가림.
- 속도·메모리·첫 토큰이 17회차 수준. 에러 없음.

## ① 측정 시킬 때 (그대로 붙여넣기)

```text
nntrainer HTP(Gemma4-26B-A4B, S26U) 18회차: 새 #4428 위로 옮긴 브랜치 확인. 코드 수정 없이 측정·보고만.

빌드: fork Seunghui98/nntrainer, 브랜치 htp/kv-int8-4428 (새 브랜치). git fetch 후 git rev-parse --short HEAD 가 819034a2d 인지
확인 (다르면 측정하지 말고 해시만 보고). 바탕이 #4428의 새 head(84ed8a799)로 바뀌어 거의 모든 파일이 바뀜:
docs/gemma4-htp-user-guide.md의 빌드 절차(Applications/CausalLM/build_android.sh --htp, DSP skel은
nntrainer/tensor/htp_backend/dsp/ 아래)를 따라 처음부터 빌드하고, IDL(nntrainer/tensor/htp_backend/dsp/nntr_hvx.idl)에서
stub/skel을 다시 생성할 것. libnntr_hvx_skel.so, libnntrainer.so, libcausallm.so, nntr_causallm(그리고 빌드가 만든 다른 .so)을
모두 새로 올리고 md5를 적을 것. 빌드 절차가 예전(test/htp/…, stage.sh)과 다르면 실제로 한 명령을 그대로 적을 것.

조건: R5KL309ALRK, models/gemma4-26b-a4b-tl2-rotated, C=16, greedy, EOS 끔, 128토큰, max_seq_len 2560, 매 실행 전 35 ℃ 이하.
환경 변수 NNTR_HTP_* / NNTR_MOE_* 는 효과가 없음: pack의 nntr_config.json을 셀마다 아래처럼 고쳐서 돌릴 것
(원본은 백업, 실행마다 어떤 내용이었는지 적을 것). 모든 셀 공통으로 "htp": {"page_cache": 0}.
  default : "htp": {"page_cache": 0}                       (+ B에서는 "profile": 1)
  off     : "htp": {"page_cache": 0, "kv_adopt": false}, "moe": {"warm_experts": 0}   (+ B에서는 "profile": 1)
pack에 이미 "htp" / "moe" 객체가 있으면 그 안에 키만 더할 것. 알 수 없는 키는 로드 때 에러가 나므로, 에러가 나면 원문을 적을 것.

## 0. 먼저 확인 (1회)
p1k를 default로 1회 ("profile": 1). run 요약 줄, pool_sync 줄 원문, "attn_m1: registered" 두 줄, 생성 텍스트 앞부분.
- 기대: prefill_seeds=30, prefill_fills=0, sliding 층 max_seq=1280.
- "FATAL ERROR" / "without seeding its decode cache" / "fell off the int8 KV path" / "could not seed" /
  "could not take its rows" / "not seeded" / "not filled" / "Unknown runtime configuration" / FastRPC·DSP 에러 / crash / hang
  이 나오면 A~C 하지 말고 그 메시지 원문과 직전 30줄, 명령, 쓴 nntr_config.json 내용만 보고.
문제없으면 A로 진행.

## A. 정확도 (profile 끔)
13회차 acc/ 의 cpu.ids 그대로. 13개 prompt (a01~a11, p1k, p2k)마다 default 1회:
  NNTR_PPL_DECODE=$Q/<prompt>.cpu.ids → r18/logs/<prompt>__default.log
prompt마다: "[PPL] decode tokens=" 줄 원문, step 줄이 15회차 int8only(r15/logs/<prompt>__int8only.log)와 같은지
(같음 / 다른 줄 수와 처음 다른 step).
- 13개 모두 같으면 A 끝.
- 하나라도 다르면: 같은 13개를 off로도 1회씩 돌려 → r18/logs/<prompt>__off.log, step 줄이 13회차 off와 같은지 적고,
  13회차 <prompt>__ref.log 를 복사한 뒤 PC에서
    python3 -I <AI-Study>/NPU/ppl_eval.py r18/logs off default
  출력 원문 전체.

## B. 속도·메모리 ("profile": 1, NNTR_PPL_DECODE 없이)
p1k · p2k, default · off 각 3회, 셀 순서 번갈아.
실행마다: prefill TPS, generation TPS, first_token ms, timeline 줄, pool_sync 줄(prefill_seeds / prefill_fills / mem_available_mib),
token1 us / misses, warm registered / cancelled, peak MiB, peak by phase (load / prefill / decode), ids md5 (17회차 같은 셀과 같은지).
중앙값 표, 17회차·15회차와 비교표 (17회차 int8only: prefill TPS 600.9 / 666.2, first_token 156.3 / 161.3 ms, peak 3520 / 3595 MiB;
15회차 off: prefill 584.1 / 651.8, first_token 396.1 / 474.4 ms, peak 3626 / 3636 MiB).

## C. chunk 경로 (1회)
p2k를 default에 NNTR_PREFILL_CHUNK=512 로 1회 ("profile": 1). pool_sync 줄 원문 (기대: prefill_seeds=120, prefill_fills=90,
prefill_fill_ms 100 ms 안팎), prefill TPS, 에러 여부.

## (선택) D. LFM2
LFM2 HTP pack이 기기에 있으면, docs/lfm2-htp-user-guide.md 대로 긴 prompt(1회 prefill에 chunk가 나뉘는 길이) 1회.
에러 여부와 run 요약 줄만. 없으면 "pack 없음".

## 규칙
- 실제로 돌린 것만 "확인됨". 못 돌린 항목은 못 돌렸다고 적을 것.
- 숫자를 추정하거나 지난 결과로 채우지 말 것.
- 실패하면 고치려 하지 말고 에러와 명령을 그대로 보고할 것. A~D 중 하나가 에러로 멈추면 그 항목만 멈추고
  나머지는 진행하되, 에러 원문과 직전 30줄을 적을 것.
- 요청서와 다르게 한 것(특히 빌드 명령, 설정 파일 내용)이 있으면 "이상한 점"에 이유와 함께 적을 것.
```

## ② 결과 정리시킬 때

```text
18회차 결과를 0, A, B, C, D 형식 그대로 정리해줘. 맨 앞에 환경(빌드 해시, 실제 빌드 명령, 새로 올린 파일과 md5, 기기, 모델,
셀별 nntr_config.json의 htp/moe 내용, 온도, 날짜), 0은 확인 결과와 원문 줄, A는 prompt별 표(prompt | default nll_sum / top1 |
15회차 int8only와 같음?) (+ off를 돌렸으면 ppl_eval 출력 원문), B는 실행별 표 + 중앙값 표 + 17·15회차 비교표, C·D는 원문 줄,
맨 뒤에 "이상한 점"과 원본 로그 경로.
```

## 결과 해석

- **A 13개 모두 15회차 int8only와 같음**: 옮긴 빌드의 계산이 그대로. 머지 가능.
- **A가 다르고 off도 13회차 off와 다름**: #4428 쪽 변경(우리 작업 밖)이 출력에 영향 → ppl_eval로 off 대비 정확도만 확인.
- **A가 다르고 off는 같음**: 옮기는 과정에서 우리 경로가 바뀐 것 → 처음 다른 prompt와 step으로 원인 추적.
- **B**: 17회차 수준이면 성능 유지. off가 15회차보다 크게 다르면 #4428 쪽 기본값 변화(설정 기본값 등)를 먼저 의심.
