# Quantized decode

`MYNAH_QUANT=f16|int8|int4` converts the decode weights once at load and keeps
them in a name-keyed cache.

Prefill is untouched: above 16 rows the call falls back to the exact f32 BLAS
path, so only the single-row autoregressive projections are quantized.

The decoder and the local transformer both route their projections through this
cache. Until they did, only the local output projection was quantized — roughly
a tenth of the decode weight traffic — which is why `int8` used to be no faster
than `f32`.

## Trade-offs

Measured on M1, 6.0 s utterance, default threads:

| mode | RTF | duration | LTAS corr vs f32 |
|---|---|---|---|
| f32 | 0.532 | 5.99 s | 1.000 |
| f16 | 0.376 | 6.22 s | 0.979 |
| int8 | 0.243 | 6.13 s | 0.973 |

**No quantized mode is a transparent replacement for f32.** All of them perturb
the greedy argmax enough to flip a near-tie token, and over ~65 autoregressive
frames that shifts EOS, so the duration moves by a few percent.

The output stays valid speech in the same voice — long-term average spectrum
correlates at 0.97–0.98, RMS and peak match — but it is a **different
realization** of the utterance, not the same waveform computed faster.

`f32` remains the default and is bit-exact: routing it through the cache is a
no-op, verified by an unchanged WAV md5.

## Choosing a mode

- **f16** is the most accurate by a wide margin. On Magpie weights its mean
  relative error is 1.8e-4 against 1.7e-2 for int8 (~96× tighter), and the
  largest weight is ~6 against the 65504 f16 limit, so overflow is not a
  concern. It keeps the activation in f32 and needs no scales.
- **int8** is the fastest.
- **int4** is available but coarser; measure before using it.

Prefer f16 when duration drift matters, int8 when speed does.

## How to judge quantized output

Do **not** use sample-wise correlation. Once the greedy trajectory diverges the
two waveforms are different realizations and correlation collapses — f16 scores
0.16 against f32 — which says nothing about quality.

Use time-alignment-insensitive measures instead:

- long-term average spectrum correlation;
- RMS and peak;
- duration within a plausible band;
- all samples finite.

## Implementation notes

Weights are stored per type: int8 with per-row symmetric scales, int4 as
Q4_0-style per-group-of-32 nibbles, f16 as IEEE half with no scales. Each kernel
keeps four output rows in flight so the activation is read once per four weight
rows.

The row range is split across the thread pool on blocks aligned to that four-row
unroll, which is bit-exact against a single serial call — a self-test checks
exactly that, for every type. Targets without NEON half converts refuse the f16
tensor and fall back to exact f32, the same contract int4 uses for a shape it
cannot represent.
