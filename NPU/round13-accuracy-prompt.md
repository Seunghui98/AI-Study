# 13회차 측정 요청: 정확도 검증 강화 (prompt 13개, off ↔ default) + 첫 토큰 확인

목적: KV adoption(default)이 원래 동작(off)보다 정확도가 나쁘지 않은지, 2개 prompt보다 훨씬 좁은 신뢰구간으로 판정.
기준: CPU attention(fp16 KV) 경로의 greedy 출력 128 토큰. 두 셀 모두 이 토큰을 강제로 넣으며 단계마다 nll을 잼.
판정: `ppl_eval.py`가 prompt 단위로 다시 뽑는 bootstrap 95 % 신뢰구간의 위 끝이 1.03 이하이고 top1 일치가 1 %p 이상 낮지
않으면 "default not worse".

## ① 측정 시킬 때 (그대로 붙여넣기)

```text
nntrainer HTP(Gemma4-26B-A4B, S26U) 13회차: 정확도 검증. 코드 수정 없이 측정·보고만.
빌드: fork Seunghui98/nntrainer, 브랜치 htp/first-token-kv-int8 (htp/accuracy-gate도 같은 커밋).
먼저 git fetch 후 git rev-parse --short HEAD 가 18ace9d69 인지 확인. 다르면 측정하지 말고 해시만 보고.
다시 빌드·stage: 이번에는 IDL이 바뀜 (graph_set_params 메서드 추가) → IDL에서 stub/skel을 다시 생성하는 전체 빌드로
skel, libnntrainer.so, libcausallm.so 모두 새로 올릴 것. stage.sh가 libc++_shared 단계에서 멈추면 12회차처럼 md5만 확인.
기기·모델·조건: R5KL309ALRK (ANDROID_SERIAL), models/gemma4-26b-a4b-tl2-rotated, C=16, greedy, EOS 끔, 128토큰,
max_seq_len 2560, NNTR_HTP_PAGE_CACHE=0, 매 실행 전 35 ℃ 이하. run.sh로. NNTR_HTP_PROFILE은 끔(정확도만).

prompt: AI-Study 저장소 NPU/accuracy_prompts/ 의 a01 ~ a11 (11개) + 기존 p1k, p2k = 13개.
Q=/data/local/tmp/nntrainer/causallm/claude_kv/acc  (새 디렉터리. 기존 $P의 파일은 지우거나 덮어쓰지 말 것)
LOGDIR=PC의 r13/logs (실행마다 stderr+stdout 전체를 <prompt>__<cell>.log 로 저장)

## 1. 기준 토큰열 (prompt마다 1회)
4·6회차에서 cpu.ids를 만들 때 쓴 fp16 pack(attention_kv_dtype이 빈 symlink pack)과 NNTR_HTP_ATTN_F16=0 (CPU attention).
- a01 ~ a11: NNTR_PPL_DECODE=$Q/<prompt>.cpu.ids (파일이 없어야 함: self 실행이 그 파일을 씀) → <prompt>__ref.log
- p1k, p2k: 기존 $P/<prompt>.cpu.ids 를 $Q 로 복사한 뒤 같은 설정으로 forced 실행 → <prompt>__ref.log
  (자기 자신의 토큰을 강제로 넣으므로 self와 같은 결과이고, 새 margin 열이 찍힘)
- 각 ref 로그의 "[PPL] decode tokens=" 줄에서 top1=128/128 인지 확인 (아니면 이상한 점에)

## 2. 두 셀 (prompt마다 각 1회, 원래 pack)
  off       NNTR_HTP_KV_ADOPT=0 NNTR_MOE_WARM=0 NNTR_PPL_DECODE=$Q/<prompt>.cpu.ids → <prompt>__off.log
  default   NNTR_PPL_DECODE=$Q/<prompt>.cpu.ids                                     → <prompt>__default.log
prompt 순서는 섞어서(off와 default를 번갈아), 같은 prompt의 두 셀은 연달아.

## 3. 보고
- prompt마다: prefill 토큰 수, ref / off / default 의 "[PPL] decode tokens=" 줄 원문, default의 prefill_seeds 값
- PC에서: python3 ppl_eval.py r13/logs off default   출력 전체
- 같은 디렉터리로: python3 ppl_compare.py <prompt>__off.log <prompt>__default.log  (prompt마다, 출력 전체)
- 생성 텍스트는 필요 없음 (forced 실행이라 기준 토큰열과 같음)
- prompt가 잘렸다는 줄(truncated / does not fit)이 있으면 원문

## 4. 첫 토큰 확인 (같은 빌드, p1k · p2k, 셀마다 5회, NNTR_HTP_PROFILE=1, NNTR_PPL_DECODE 없이, 셀 순서 돌아가며)
셀:  default (스위치 없음) / nobatch (NNTR_HTP_PARAM_BATCH=0) / off (NNTR_HTP_KV_ADOPT=0 NNTR_MOE_WARM=0)
실행마다: prefill TPS, generation TPS, first_token ms, timeline 줄 (graph_init / to_token / pool_sync / to_packet / packet),
params 줄 (params / mib / batched / ms / dsp_ms), token1 us / misses, warm registered / cancelled, peak MiB,
peak by phase, ids md5가 12회차 같은 동작 셀과 같은지.
중앙값 표, 같은 rep 차이표 (default − nobatch의 params ms · first_token, default − off의 first_token · peak).
12회차 중앙값 (default / off): first_token 170.3 / 427.8 · 175.5 / 498.6 ms, to_token 35.4 / 138.1 · 39.1 / 209.3 ms(rep 중앙),
params 38.3 / 43.9 · 38.8 / 44.7 ms.

## 규칙
- 실제로 돌린 것만 "확인됨". 못 돌린 항목은 못 돌렸다고 적을 것.
- 숫자를 추정하거나 지난 결과로 채우지 말 것.
- 실패하면 고치려 하지 말고 에러와 명령을 그대로 보고할 것.
- 요청서와 다르게 한 것이 있으면 "이상한 점"에 이유와 함께 적을 것.
```

## ② 결과 정리시킬 때 (측정 세션에)

```text
13회차 결과를 요청서의 3.과 4. 형식 그대로 정리해줘 (4.는 실행별 표 + 중앙값 표 + 같은 rep 차이표 + 12회차와 비교표). 맨 앞에 0. 환경(빌드 해시, 새로 올린 파일과 md5, 기기, 모델, 온도, 날짜),
그다음 prompt별 표(prompt | prefill 토큰 | ref nll_sum | off nll_sum / top1 | default nll_sum / top1 | prefill_seeds),
ppl_eval.py 출력 원문 전체, prompt별 ppl_compare.py 출력 원문 전체, 맨 뒤에 "이상한 점"과 원본 로그 경로.
```

## 결과 해석 (4. 첫 토큰)

- **to_token이 ~35 ms → 수 ms**: 층마다 RoPE 표를 새로 만들고 memcmp하던 비용이 원인이었음 (첫 토큰 −30 ms 안팎 기대).
- **params ms가 nobatch보다 작으면**: 301번의 왕복이 원인이었음 (dsp_ms 17.5 ms 근처까지 내려갈 수 있음).
- ids는 12회차와 같아야 함 (표와 파라미터 값은 그대로, 보내는 방식만 바뀜).

## 결과 해석 (정확도)

- **default not worse**: KV adoption 기본값 유지.
- **undecided**: 신뢰구간이 여전히 넓음 → prompt를 늘리거나 256 토큰으로 (num_to_generate 변경).
- **default worse**: 기본값을 다시 끄고(한 줄) 원인 분석. 갈라지는 단계의 margin/gap 줄로 어느 층·위치에서 차이가 나는지 좁힘.
- margin 요약: default가 기준과 다르게 고른 단계에서 기준의 margin이 대부분 0.5 미만이면 동점 근처에서 뒤집힌 것 (무해),
  margin이 큰데 다르게 골랐다면 실제 오차.
