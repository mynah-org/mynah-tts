# E10-5 — int8 for the SEANet conv stack: what it bought, and what it cost

Item: `PLAN.md` E10-5.  Files: `src/convq8.{c,h}` (new), `src/qmat.c`
(exported primitives + a vectorised activation quantizer), `src/seanet.{c,h}`,
`src/engine_pocket.c`, `tests/codec_int8_quality.py`.

All numbers on this Mac, `BLAS=none`, `MYNAH_THREADS=2`, `models/pocket-en`,
seed 1 — a **development signal, not a product claim** (CLAUDE.md).  The
measurement rule from [`measuring-the-codec.md`](measuring-the-codec.md) is
followed: everything here is `BLAS=none`, because that is what Linux ships and
because the Accelerate numbers put the phases in a different order.

## Problem

`codec.conv_stack` is a quarter of the wall and, in the shipping
configuration, 81.5% of it is two GEMM families.  The conv1d half — 42% — had
**no quantized kernel at all**: `codec_conv` in the default spec resolved to
`mimi.quantizer.output_proj [512][32]`, one matvec, so a spec that said
`codec_conv:int8` was quantizing about 0.03% of what its name covers.

## What was built

`src/convq8.c`: an int8 tap GEMM with its own weight memo, keyed the way
`sea_taps_all()` is keyed, because this layer never sees a tensor name.  The
int8 arithmetic itself is **qmat's, exported** (`mynah_qmat_pack_q8`,
`mynah_qmat_act_quantize`, `mynah_qmat_dots_i8`) rather than rewritten: the
activation encoding is a property of the host — signed for SDOT, unsigned
x+128 with a row-sum correction for VPDPBUSD — and a second copy of that
dispatch would have silently given up VNNI on the x86 half of production.

The three transposed convolutions are **not** included, per the item: the
reference measured its int8 convtranspose slower than f32 sgemm and ships it
off.  They are 48% of the region and are the obvious next question — see the
open item at the bottom.

## Per-shape, against the f32 path it replaces

The eight conv shapes one utterance runs, `MYNAH_CONVQ8_PROFILE=1` against the
f32 `[SGEMM-SHAPES]` totals for the same run:

| m | n | k | taps | f32 ms | int8 ms | ratio | in the gate? |
|---|---|---|---|---|---|---|---|
| 512 | 16 | 512 | 7 | 35.39 | 8.40 | **4.21x** | yes |
| 128 | 96 | 256 | 3 | 10.02 | 3.44 | **2.91x** | yes |
| 256 | 96 | 128 | 1 | 2.88 | 1.85 | 1.56x | yes |
| 64 | 480 | 128 | 3 | 11.53 | 8.16 | 1.41x | yes |
| 32 | 1920 | 64 | 3 | 19.63 | 13.62 | 1.44x | **no — quality** |
| 128 | 480 | 64 | 1 | 3.27 | 3.89 | 0.84x | no |
| 64 | 1920 | 32 | 1 | 2.89 | 7.67 | 0.38x | no |
| 1 | 1920 | 64 | 3 | 1.25 | 4.46 | 0.28x | no |

The gate is `k * taps >= 128 && m >= 64`, and each half is one of those sign
changes:

* **`k * taps`** is the depth of one output element's accumulation.  Below 128
  the SDOT kernel cannot amortise its per-row setup and the f32 panel kernel,
  which keeps a whole C tile in registers instead, wins.  The sign changes
  between 128 (1.56x) and 64 (0.84x).
* **`m`** is the output channel count, and it is doing two jobs.  Speed: the
  activation pass is paid once per call whatever `m` is, so at `m = 1` it *is*
  the call — the last convolution spends 64% of its time quantizing for a
  single output row.  **Quality**: a SEANet decoder narrows toward the
  waveform, so the narrow stages are the LAST ones and their residual reaches
  the output with nothing downstream to average it.

## The quality finding, which decided the gate

Admitting one more stage at a time, same utterance:

| stages int8 | SNR | waveform corr | **log-mel corr** |
|---|---|---|---|
| entry conv only | 45.0 dB | 0.999984 | 0.999957 |
| + k*taps >= 384 | 38.7 dB | 0.999932 | 0.997576 |
| + the 32-channel stage | 36.1 dB | 0.999877 | **0.978535** |
| + k*taps >= 128 | 35.1 dB | 0.999846 | 0.978447 |

The 32-channel stage costs **8x on log-mel** to save 6 ms of 44.  Refused.

**The two measures disagree on purpose, and that is the point.** The int8 this
project already ships in the same region — `codec_transformer:int8` — moves
this utterance by 25.3 dB SNR and **0.99852** waveform correlation but
**0.99953** log-mel: it produces a *different but spectrally identical*
waveform.  The conv stack's int8 does the opposite — 0.9999 waveform
correlation, a *lower* log-mel — because it adds a low-level broadband
residual rather than moving the signal.  Judging this change on the waveform
number alone would have read "ten times better than what already ships" and
would have been wrong.  One number could not have caught it.

Shipped configuration, three texts x three seeds: waveform correlation
0.999886-0.999920, log-mel 0.996234-0.997161, SNR 36.4-37.9 dB, and **the
sample count identical in every pair** — the frame count and the EOS step do
not move, which is the property int8 was accepted on in the first place.
Gated by `make codec-int8-quality MODEL_DIR=...`.

## Result

Paired interleaved, five rounds each, medians:

| | f32 | int8 | ratio |
|---|---|---|---|
| `codec.conv_stack` | 204.2 ms | 170.0 ms | **1.20x** (5/5 rounds) |
| whole request | 0.596 s | 0.560 s | **1.06x** (5/5 rounds) |

The conv GEMM family itself goes 86.9 -> ~48.8 ms.  The region only moves 1.20x
because `convtr.gemm` — which this item deliberately does not touch — is now
**57% of what is left**.

## Four things found by measuring, not by reading

1. **The activation quantizer was scalar, and it was 40.8% of the int8 path** —
   more than the SDOT it feeds.  Vectorised in `src/qmat.c` (NEON `vcvta`,
   which rounds ties away from zero exactly as the scalar `v + 0.5f` does):
   30.5 -> 10.7 ms, and the decode loop gets it too.  Asserted **byte
   identical** to the scalar reference, both encodings, in
   `self_test_act_quantize()`.
2. **Tiling the activation transpose is slower** (30.5 -> 34.1 ms).  It reads
   each cache line once, which is the right instinct, and pays for it by
   writing with the stride instead of reading with it.  The cost was never the
   memory pattern; it was (1).  Reverted, and the finding is in the comment so
   the next person does not re-try it.
3. **The memo keyed on a weight POINTER, and malloc recycles addresses.**  The
   gate self-test freed one tensor and allocated another of the same shape at
   the same address and got the first one's quantized bytes: relative error
   0.0046 -> 1.45.  Fixed with a 64-sample content fingerprint in the key.
   **`sea_taps_all()` in `src/seanet.c` has the same hazard** and is not fixed
   here — see `PLAN.md` E10-13.
4. **My own error path had a use-after-free**: it read `ref[r]` after
   `free(ref)`, so the first mutation run reported the reference value as "0" —
   a diagnostic lying about exactly the number it exists to print.

## What the gates actually catch

Every mutation below was applied, built, and the named gate observed to fail:

| mutation | caught by |
|---|---|
| last NEON lane of `mynah_qmat_dots_i8` | batch-width identity (`==`, not a bound) |
| a tap dropped from the sum | relative L2, 0.87 against a 0.0060 bound |
| tap column off by one | relative L2, 1.30 |
| per-row scale replaced by row 0's | block-independence (`memcmp`) |
| `vcvtn` (ties to even) for `vcvta` | `self_test_act_quantize`, byte compare |
| absmax tail dropped | `self_test_act_quantize`, scale compare |
| u8 `+128` bias dropped | `self_test_act_quantize` |
| **rowsum dropped in the unsigned path** | **only under `MYNAH_QMAT_VNNI=scalar`** |
| **a lane of the 4-row unsigned block** | **only under `MYNAH_QMAT_VNNI=scalar`** |

The last two are the x86 encoding, which on an ARM machine is compiled,
shipped and never executed.  `MYNAH_QMAT_VNNI=scalar` forces the portable
unsigned kernel on any host and **had no gate using it** — the level existed
for exactly this purpose.  `make self-test` and `make qmat-test` now run it.

The bounds are **absolute**, not relative to a baseline: E10-6 found twice that
an ordering-only gate passes mutations because a symmetric degradation moves
the subject and the baseline together.  `MYNAH_CONVQ8_DEBUG=1` reprints the
measured column so a bound is never a number someone remembered.

## Open, and what would close it

* **The transposed convolutions are 57% of what is left.**  The item says skip
  them on the reference's measurement, and that measurement was of *their*
  kernel.  Ours gets 4.21x on `512x16x512x7`, and `convtr`'s biggest shape is
  `3072x16x512` — the same k, six times the m, which is where this kernel is
  strongest.  Worth re-deciding with our own number rather than inheriting
  theirs.  `PLAN.md` E10-14.
* **The two thresholds are from this machine.**  Their *shape* is structural —
  too shallow to amortise a kernel, too close to the output to hide a residual
  — but the constants are not.  Re-take the per-shape table on the Linux box;
  it is eight numbers and one command.
* **The x86 kernel is unexecuted**, as ever.  `MYNAH_QMAT_VNNI=scalar` now
  covers its *algebra* on any host; it does not cover VPDPBUSD itself.
