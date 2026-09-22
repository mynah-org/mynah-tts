#!/bin/sh
# Does changing the kernel tier change the AUDIO, and does each control vary
# exactly one thing?
#
#   MODEL_DIR=models/pocket-en sh tests/x86_tier_parity.sh
#
# WHY IT EXISTS. x86 now picks four kernel families at runtime, and the only
# host that can execute all of them is a rented one. Correctness across those
# tiers is not something the unit tests can state: they compare kernels on
# fixed buffers, and this engine is autoregressive, so a one-ULP disagreement
# is amplified by every subsequent step. The question is whether the AUDIO
# survives, and it can only be asked where the tiers exist.
#
# TWO CHECKS, AND THE SECOND IS THE ONE THAT EARNED ITS PLACE.
#
#   1. Tiers whose arithmetic is EXACT must produce BYTE-IDENTICAL audio. The
#      int8 dot is exact int32 by construction, so avx512bw and avx2 -- the same
#      signed algebra at two widths -- may not differ by one sample.
#   2. Every forcing must move exactly the dispatch rows it is supposed to
#      move. On 2026-09-22 MYNAH_QMAT_VNNI=scalar turned out to ALSO flip
#      codec.conv_int8_host from eligible to off, because the codec asks which
#      int8 kernel RESOLVED and reads a forced scalar as a host without a dot
#      unit. That is defensible behaviour and the dispatch map states it -- and
#      it still made an A/B that varied two things, which is how an audio
#      difference gets blamed on a kernel that did not cause it. A control that
#      moves two rows is not a control.
set -u
cd "$(dirname "$0")/.."
M="${MODEL_DIR:?set MODEL_DIR to a Pocket pack}"
BIN="${BIN:-build/cpu/mynah-tts}"
OUT="${OUT:-build/tier-parity}"
TEXT="${TEXT:-The quick brown fox jumps over the lazy dog.}"
STEPS="${STEPS:-40}"

case "$(uname -m)" in
  x86_64|amd64) ;;
  *) echo "SKIP x86-tier-parity: not an x86 host, there are no tiers to compare"; exit 0 ;;
esac
[ -x "$BIN" ] || { echo "SKIP x86-tier-parity: no $BIN"; exit 0; }

rm -rf "$OUT"; mkdir -p "$OUT"
fail=0
say() { printf '  %-5s %s\n' "$1" "$2"; [ "$1" = "ok" ] || fail=1; }

# temperature 0: generation is deterministic, so any difference is arithmetic
# and not the sampler. Verified by the baseline-twice check below, which is not
# ceremony -- without it every other line here could be noise.
synth() {
    n=$1; shift
    env "$@" "$BIN" --synthesize "$M" --text "$TEXT" --lang en --speaker 0 \
        --max-steps "$STEPS" --seed 42 --temperature 0 \
        --output "$OUT/$n.wav" >"$OUT/$n.log" 2>&1 \
      || { say FAIL "$n: synthesis failed -- $(tail -1 "$OUT/$n.log")"; return 1; }
    # pool.* is excluded and that is not laziness: pool.spin is a MEASURED
    # per-host calibration (a time, converted to an iteration count from a
    # sampled ns/relax) and it moves between two runs of the same binary. The
    # first version of this test compared it and reported the baseline as
    # differing from itself.
    env "$@" "$BIN" --dispatch-map 2>/dev/null \
      | awk '$1 ~ /^(isa|quant|sgemm|codec|kernel|cpu)\./ {print $1, $5}' \
      > "$OUT/$n.rows"
}

rows_moved() { diff "$OUT/baseline.rows" "$OUT/$1.rows" 2>/dev/null \
                 | awk '/^[<>]/{print $2}' | sort -u | tr '\n' ' '; }

synth baseline  || exit 1
synth baseline2 || exit 1
synth vnni_off    MYNAH_QMAT_VNNI=off
synth avx512_off  MYNAH_QMAT_VNNI=off MYNAH_QMAT_AVX512=off
synth bf16_off    MYNAH_QMAT_BF16DOT=off
synth f32_scalar  MYNAH_KERNELS_X86=scalar

echo "== determinism, without which nothing below means anything =="
cmp -s "$OUT/baseline.wav" "$OUT/baseline2.wav" \
  && say ok "the same tier twice is byte-identical" \
  || say FAIL "the same tier twice DIFFERS -- this engine is not deterministic here"

echo "== exact arithmetic must give identical audio =="
if [ -f "$OUT/avx512_off.wav" ]; then
  cmp -s "$OUT/vnni_off.wav" "$OUT/avx512_off.wav" \
    && say ok "int8 avx512bw == avx2: byte-identical, as exact int32 requires" \
    || say FAIL "int8 avx512bw != avx2, and the int32 dot is exact by construction"
fi

echo "== inexact tiers: the utterance must survive, not the bits =="
for n in bf16_off f32_scalar vnni_off; do
  a=$(stat -c%s "$OUT/baseline.wav" 2>/dev/null || echo 0)
  b=$(stat -c%s "$OUT/$n.wav" 2>/dev/null || echo 0)
  if [ "$a" = "$b" ] && [ "$a" != "0" ]; then
    say ok "$n: same length ($a B) -- the AR loop made the same decisions"
  else
    say FAIL "$n: length $b against $a -- a different number of frames was emitted"
  fi
done

echo "== each control must move only rows it is ALLOWED to move =="
# An allowlist, not an exact set, and the difference matters twice over.
#
# A row may legitimately not move: on a SIMD=auto build BOTH sgemm variants are
# AVX2, so MYNAH_KERNELS_X86=scalar cannot change sgemm.kernel -- the honest
# f32 comparison there is between BUILDS, which is what the micro-bench says
# too. Requiring it would fail a correct binary.
#
# And codec.conv_int8_host IS allowed to move with the int8 forcings, because
# it genuinely does: src/convq8.c asks which int8 kernel RESOLVED and reads a
# forced-down one as a host without a dot unit, so the conv stack stays f32.
# That behaviour is deliberate and the dispatch map states it. It is listed
# here so it stops being a surprise -- the point of this check is that a
# control cannot move something NOT on its list, which is how an audio
# difference gets blamed on the wrong kernel.
check_rows() {
  n=$1; allow=$2
  bad=""
  for r in $(rows_moved "$n"); do
    case " $allow " in *" $r "*) ;; *) bad="$bad $r";; esac
  done
  if [ -z "$bad" ]; then
    say ok "$n moves only allowed rows: $(rows_moved "$n" | sed 's/ *$//')"
  else
    say FAIL "$n also moved:$bad (allowed: $allow)"
  fi
}
check_rows baseline2   ""
# vnni_off carries MORE rows than avx512_off, which looks backwards and is
# correct: turning VNNI off leaves AVX-512 F/BW/VL standing, so the E14-2
# avx512bw int8 tier WINS and its two rows flip ON. Turning AVX-512 off as well
# removes that winner, so they stay OFF and never move. The rows mean "this is
# the kernel that RUNS" since 455e1f2, and this list was written before they
# did -- the run that caught it was the first with a model pack on a host that
# has both units.
check_rows vnni_off    "isa.x86.avx512vnni isa.x86.avx512f isa.x86.avx512bw quant.int8_kernel codec.conv_int8_host"
check_rows avx512_off  "isa.x86.avx512vnni isa.x86.avx512f isa.x86.avx512bw isa.x86.avx512vl quant.int8_kernel codec.conv_int8_host"
check_rows bf16_off    "isa.arm.bf16 isa.x86.avx512bf16"
check_rows f32_scalar  "isa.x86.avx2 isa.x86.fma sgemm.kernel"

echo
[ "$fail" -eq 0 ] && echo "x86-tier-parity: PASS" || echo "x86-tier-parity: FAIL"
exit "$fail"
