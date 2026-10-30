#!/usr/bin/env bash
# First-token / per-dim K scale device runs (first-token-device-test.md).
#
# Usage (from the nntrainer checkout, after build_android.sh --htp --app):
#   bash run_first_token_tests.sh stage            # push binaries once
#   bash run_first_token_tests.sh run  p1024.txt   # the switch matrix, 3 reps
#   bash run_first_token_tests.sh run  p2048.txt
#   bash run_first_token_tests.sh summary          # grep the numbers
#
# Override: D (device dir), PACK (model pack on the device), REPS, OUT,
# CASES_ONLY (a space-separated subset of the case names below).
set -eu
: "${D:=/data/local/tmp/htp}"
: "${PACK:=/sdcard/Download/aistudio-mobile/models/gemma4-26b-a4b-htp-w4a32}"
# only these cases when set, e.g. CASES_ONLY="base warm8 warm16"
: "${CASES_ONLY:=}"
: "${REPS:=3}"
: "${OUT:=ft_logs}"

# name : environment (on top of the defaults built into the library)
CASES=(
  "base:"
  "kdim:NNTR_HTP_Q2_KDIM=1"
  "kdim_seedq:NNTR_HTP_Q2_KDIM=1 NNTR_HTP_KV_SEED_Q=1"
  "seedq_only:NNTR_HTP_KV_SEED_Q=1"
  "kdim_f32check:NNTR_HTP_Q2_KDIM=1 NNTR_HTP_ATTN_F32_CHECK=1 NNTR_HTP_ATTN_TRACE=1"
  "kdim_hostcalib:NNTR_HTP_Q2_KDIM=1 NNTR_HTP_ATTN_DSP_CALIB=0"
  "warm8:NNTR_MOE_WARM=8"
  "warm16:NNTR_MOE_WARM=16"
)

stage() {
  : "${ANDROID_NDK:?export ANDROID_NDK}" "${HEXKL_ROOT:?export HEXKL_ROOT}"
  adb shell mkdir -p "$D"
  for f in builddir_app/htp/nntr_causallm builddir_app/htp/libcausallm.so \
           builddir_app/htp/libquick_dot_ai_api.so \
           builddir/android_build_result/lib/arm64-v8a/libnntrainer.so \
           builddir/android_build_result/lib/arm64-v8a/libccapi-nntrainer.so \
           "$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" \
           test/htp/build/libnntr_hvx_skel.so \
           "$HEXKL_ROOT/lib/${HEXKL_SDK_VER:-6.4.0.2}/armv8_android26/libsdkl.so"; do
    [ -f "$f" ] || { echo "missing: $f" >&2; exit 1; }
    adb push "$f" "$D/" >/dev/null
  done
  adb shell chmod 755 "$D/nntr_causallm"
  echo "staged in $D ($(git rev-parse --short HEAD))"
}

run() {
  local prompt=$1 tag
  tag=$(basename "$prompt" .txt)
  adb push "$prompt" "$D/prompt_$tag.txt" >/dev/null
  mkdir -p "$OUT"
  for c in "${CASES[@]}"; do
    local name=${c%%:*}
    local envs=${c#*:}
    if [ -n "$CASES_ONLY" ] && ! printf ' %s ' $CASES_ONLY | grep -q " $name "; then
      continue
    fi
    for r in $(seq 1 "$REPS"); do
      local log="$OUT/${tag}_${name}_r$r.log"
      echo "== $tag $name rep $r"
      adb shell "cd $D && LD_LIBRARY_PATH=$D ADSP_LIBRARY_PATH=$D \
        NNTR_HTP_PROFILE=1 NNTR_HTP_TOKEN_TRACE=$D/trace_${tag}_${name}_r$r.json \
        $envs ./nntr_causallm $PACK \"\$(cat $D/prompt_$tag.txt)\"" \
        >"$log" 2>&1 || echo "   exit $? (see $log)"
      adb pull "$D/trace_${tag}_${name}_r$r.json" "$OUT/" >/dev/null 2>&1 || true
    done
  done
}

summary() {
  for log in "$OUT"/*.log; do
    echo "== $(basename "$log" .log)"
    grep -hE "^prefill:|^generation:|^peak memory|\[HTP\] t1:|token driver: first token|token driver: token [1-4] |t1 seed pos=" "$log" | tail -n 14
    grep -hE "seeds decode from fp16|attn_m1_seed_kvq:|quantized KV cache .* failed" "$log" | sort | uniq -c | head -5
    grep -hE "f32 edge check" "$log" | awk '{for(i=1;i<=NF;i++) if($i ~ /^mismatches=/) print $i}' | sort | uniq -c | head -3
  done
  echo "== token ids vs base (same prompt, same rep)"
  for t in "$OUT"/trace_*_base_r1.json; do
    [ -f "$t" ] || continue
    local tag
    tag=$(basename "$t" | sed 's/^trace_\(.*\)_base_r1.json$/\1/')
    for c in "${CASES[@]}"; do
      local name=${c%%:*}
      local other="$OUT/trace_${tag}_${name}_r1.json"
      if [ -f "$other" ]; then
        printf '%-8s %-16s %s\n' "$tag" "$name" \
          "$(cmp -s "$t" "$other" && echo same || echo DIFFERENT)"
      fi
    done
  done
}

case "${1:-}" in
  stage) stage ;;
  run) run "${2:?prompt file}" ;;
  summary) summary ;;
  *) sed -n 2,10p "$0"; exit 1 ;;
esac
