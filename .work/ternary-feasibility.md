# E11 — Is PocketTTS ternarizable? (W1.58 post-training)

Status: **IN PROGRESS** — opened 2026-09-20, measured the same day. Tool: `tools/ternary_feasibility.py`.
Report: `docs/ternary-feasibility.md`. Results land in `build/ternary/*.json`.

## The question, and why it is asked in this order

Ternary weights would be the largest single change to `src/qmat.c` since it was
written: a new qtype, 2-bit packing, a per-row scale epilogue, SDOT/SMMLA and
VNNI accumulation paths, a packer, a cache, a dispatch row and a self-test on
both ISAs. That is the expensive half. The cheap half is a fake-quantization
sweep in Python that asks whether the *trained checkpoint* tolerates ternary at
all.

**This item is the cheap half only. No C, no SIMD, no kernel.** If the answer is
no, or "only for 30% of the weights", we will have spent a day instead of two
weeks, and the negative result is the deliverable.

The repo has been wrong in this exact direction before. `PLAN.md` recorded
"int8 breaks PocketTTS parity" as a fact for weeks; it was a category error —
f32 against f32 with a different seed gives hidden `rel_l2` 1.05-1.07, twenty
times the 5.9e-2 that had been flagged. Weight-space MSE is a proxy, and it has
already lied to us once. So the gate here is audio, and the ranking metric is
output error on real activations, not `||W - Ŵ||`.

## Prior art read before designing the experiment

| Paper | What we took | What we did not take |
|---|---|---|
| PT²-LLM (ICLR 2026, [2510.03267](https://arxiv.org/abs/2510.03267)) | ITF — alternating closed-form (α, μ) against flexible rounding of T; AGA — refit the grid under the calibration metric with T frozen | SSR (structural-similarity column reordering); we use `diag(H)` ordering instead, which is GPTQ's act-order |
| TWLA ([2606.13054](https://arxiv.org/abs/2606.13054)) | E2M-ATQ, both stages: Euclidean warm start with residual-mean correction, then manifold relocation | KOTMS in full — we fit a Kronecker rotation by random-direction descent on the TriGMM surrogate, not Adam through the Cayley map; ILA-AMP bit allocation; W1.58A4 entirely |
| Qwen3-4B ternarization ([2609.01962](https://arxiv.org/abs/2609.01962)) | the honest bit-budget accounting (1.641 effective bits, not 1.58) and the reality check that 81.6% of parameters ternarized still cost 64.5% → 54.7% task accuracy | nothing methodological; it is the cautionary result |

**A finding from the reading itself, before any measurement**: PT²'s AGA and
TWLA's E2M-ATQ stage 2 are *the same estimator*. Both freeze T and solve a
per-row 2×2 system in the metric `S = XᵀX`; the papers differ in notation and in
how they arrive at T, not in the relocation. The tool implements them as two
methods anyway, so the gap between them measures the warm start and nothing
else. Anyone about to implement both should know they are buying one thing.

**A second finding, also free**: the Qwen3 paper does not claim a speedup. Its
Triton kernel is **4.6× slower than FP16**. Compression is not acceleration;
ternary earns its place here only if it reduces *weight traffic on the path
that is bandwidth-bound* or *raises MACs per instruction on the path that is
compute-bound*. Which of those applies is a per-tensor question, answered in §
"Where the bytes are vs where the MACs are" below.

## Method

`tools/ternary_feasibility.py`, six subcommands, phases in order:

- `census` — every tensor classified by component / group / role, with the
  canonical 2-D form `(rows, contract)` each quantizer sees and whether the
  streaming loop touches it. Nothing is ternarized by default: a tensor that
  falls through the table lands in `other` and stays FP.
- `calib` — hooks the upstream `pocket-tts` model and accumulates, per module,
  the **exact** Gram matrix `S = XᵀX` over every activation row the layer sees,
  plus a 512-row reservoir sample. Two structures because there are two
  consumers: GPTQ's Hessian inverse is meaningless on a rank-512 `S` of
  dimension 3584, and the output-error metric needs actual rows.
- `sweep` — one layer at a time, every method, weight-level and output-level
  error.
- `cumulative` — rank least-sensitive-first, ternarize a growing prefix, and
  generate real audio at each point.
- `shapes` — the M×K table a future kernel author would work from.
- `report` — regenerates the tables in `docs/ternary-feasibility.md` between
  markers; the prose around them is hand-written.

Safety contract: `Pack` is read-only and `tensor()` returns a fresh copy. The
end-to-end runs monkey-patch a throwaway upstream model in memory. **No model
pack is ever written.**

## Where the bytes are vs where the MACs are (measured, `shapes`)

Run 2026-09-20 on the shipped `models/pocket-en` (109,502,146 params, bf16,
219.0 MB). Streaming-path ternarizable weights: **24.06 MB** at 2 bits + f16
per-row scale and shift.

| | % of ternary bytes | % of MACs per frame |
|---|---|---|
| `flow_lm.transformer` (backbone, 12.5 Hz) | **72.8** | 20.1 |
| `mimi.decoder*` (200 Hz → 24 kHz) | 27.2 | **79.9** |

This is the single most important number in the item and it was free. The
backbone holds three quarters of the weight bytes and does one fifth of the
arithmetic, because it runs once per 80 ms frame; the codec holds a quarter of
the bytes and does four fifths of the arithmetic, because it runs up to 1920
times per frame. **They want opposite things from ternary.** The backbone would
be buying bandwidth — 39.8 MB of bf16 becomes 17.5 MB, which is the difference
between spilling L2 and not. The codec would be buying MAC width — SDOT/SMMLA
over 2-bit-unpacked int8 against the bf16 BFMMLA path E4-8 just shipped.

Both are plausible. Neither is proven. That is what the sweep is for.

## Acceptance gate

The item closes when `docs/ternary-feasibility.md` answers, with numbers:

1. what fraction of parameters is a genuine ternary candidate;
2. the least- and most-sensitive modules, ranked by output NMSE on real
   activations;
3. how much activation-aware PTQ beats naive ternary, per group;
4. whether PT²-style or TWLA-style wins on *this* architecture;
5. how far the cumulative curve goes before audio degrades, judged by log-mel
   correlation against the FP reference **and by listening**;
6. the best mixed-precision layout and its **real** effective bits/weight, with
   scales and metadata counted, not a nominal 1.58;
7. a go / no-go on the C kernel work, and if go, the first 3-5 shapes.

A "go" requires all three of: a layout that keeps mel correlation within the
FP-vs-FP seed noise floor, a byte reduction large enough to change the
bandwidth story on the backbone or the MAC story on the codec, and the audio
passing a listening check under the supreme rule — *if it sounds good, it wins*.

**A negative result closes this item just as well as a positive one**, and is
cheaper than the alternative, which is finding out after the kernel exists.

## Open risks

- **Sampler divergence confounds end-to-end audio.** Same seed, same text, but
  a changed weight moves the flow-head noise draw, so the two waveforms are not
  aligned sample-for-sample even when both are good. Duration ratio and
  log-mel correlation absorb this; waveform `rel` does not and is reported for
  information only. The FP-vs-FP floor must be measured first or every number
  below it is noise.
- **KOTMS costs a rotation at runtime.** `R = R₁ ⊗ R₂` is cheap (`n₁²+n₂²`
  instead of `d²`), but it is not free, and it has to be applied to the
  *activation* on every call. If KOTMS wins the sweep, the win must be re-priced
  against that cost before it counts.
- **Six calibration utterances, one voice, one language.** Enough to rank
  layers; not enough to promote a layout. If the answer is "go", the calibration
  set is the first thing to widen.

---

# Measured, 2026-09-20 (M1 Mac, `models/pocket-en`, upstream `pocket-tts`)

## 1. The metric had to be fixed before any number meant anything

First attempt, at the shipped temperature 0.3: the **FP-vs-FP floor is
mel_corr 0.648** (min 0.572). Two runs of the same sentence, same model, same
weights, different sampler seed, correlate at 0.648 — so an end-to-end metric
against an FP reference was measuring the dice, not the quantizer. The
resulting curve was non-monotonic and every point from 10% to 50% sat *above*
the floor.

At **temperature 0 the two runs are bit identical** (mel_corr 1.0000, mel_l1
0.0000, wave_rel 0.000). That is the measurement mode: the floor is exactly
zero and every remaining difference is attributable to the weights. The cost is
that 0 is not how the model ships, so the audio is flatter (rms 0.097 vs
0.133). **Measure at 0; listen at 0.3.**

Both numbers are kept, because they answer different questions. The 0.648 is
not noise to be eliminated — it is the calibration of "how different are two
equally good takes", and it is the yardstick a mel correlation has to be read
against.

## 2. Method ranking (output NMSE on real activations, streaming groups)

| group | params | int8 | int4 | naive | itf | aga | e2m | **gptq** |
|---|---|---|---|---|---|---|---|---|
| backbone/ffn_up | 25.2M | 0.0000 | 0.0017 | 0.0492 | 0.0486 | 0.0151 | 0.0151 | **0.0024** |
| backbone/ffn_down | 25.2M | 0.0004 | 0.1222 | 0.4030 | 0.2897 | 0.1673 | 0.1673 | **0.0120** |
| backbone/attn_qkv | 18.9M | 0.0000 | 0.0016 | 0.0594 | 0.0567 | 0.0132 | 0.0132 | **0.0028** |
| backbone/attn_out | 6.3M | 0.0001 | 0.0218 | 0.1772 | 0.1757 | 0.1421 | 0.1424 | **0.0088** |
| flow_head/flow_adaln | 5.2M | 0.0000 | 0.0016 | 0.0203 | 0.0084 | 0.0023 | 0.0023 | **0.0002** |
| flow_head/flow_mlp | 3.1M | 0.0001 | 0.0274 | 0.1785 | 0.1734 | 0.1063 | 0.1062 | **0.0128** |
| mimi_dec_tr/ffn_down | 2.1M | 0.0001 | 0.0290 | 0.1795 | 0.1758 | 0.1571 | 0.1572 | **0.0383** |
| mimi_decoder/conv | 2.0M | 0.0002 | 0.0391 | 0.2906 | 0.2761 | 0.1595 | 0.1561 | **0.0515** |

Four things fall out of this table:

- **Weight MSE is the wrong objective, demonstrated.** On
  `layers.0.self_attn.out_proj`, AGA makes the *weight* error worse
  (nmse 0.213 → 0.279) while halving the output error; GPTQ triples the weight
  error (0.690) and cuts the output error 27×. Any ranking built on ‖W − Ŵ‖
  would have chosen the wrong method, and this repo has already been burned once
  by trusting a weight-space proxy.
- **AGA ≈ E2M-ATQ, confirmed numerically.** Every row agrees to within 0.3%.
  They are the same estimator; the warm start buys nothing measurable here.
  **Do not budget to implement both.**
- **GPTQ-style compensation is the whole game**, worth 5–15× over ITF/AGA and
  10–30× over naive. Ternary without error compensation is not competitive.
- **2-bit with compensation beats 4-bit without.** On the backbone, GPTQ-ternary
  scores **0.0065** mean output NMSE against int4 round-to-nearest's **0.0368** —
  5.7× better at half the bits. The interesting comparison was never
  "ternary vs int8"; it is "ternary+GPTQ vs int4", and ternary wins it.
- **int8 is nearly free** (1.1e-4 mean on the backbone, 6.4e-4 worst). This is
  a result about `POCKET_QG_DEFAULT_SPEC`, not about ternary, and it is followed
  up separately.

## 3. Sensitivity ranking, least to most (GPTQ)

Least sensitive: `flow_net.time_embed` (0.0000, see the caveat below),
`flow_adaln` (0.0002), backbone `attn_qkv` (0.0028) and `ffn_up` (0.0024).

Most sensitive, and all of them tiny: `flow_net.input_proj` (16K params,
0.1875), `flow_lm.input_linear` (32K, 0.1617), `mimi.quantizer.output_proj`
(16K, 0.1580), and the Mimi decoder's 1×1 residual convs (8–32K, 0.09–0.13).
**The protected set costs ~100K parameters, under 0.1% of the model.** That is
the cheapest possible thing to protect, and it is measured rather than assumed.

Within the backbone, sensitivity rises with depth for `attn_qkv`
(0.0004 → 0.0051 from layer 0 to layer 3) and `ffn_down` is 5× more sensitive
than `ffn_up` (0.0120 vs 0.0024) despite identical parameter counts.

**Caveat on `time_embed`**: its score of exactly 0.0000 is degenerate, not
excellent. With `flow_decode_steps = 1` the timestep is fixed, so the layer sees
one activation direction, the Gram is effectively rank 1, and AGA fits that
direction exactly. It was probed on its own end to end and the result is
**mel_corr 1.0000 — bit identical output**. So the degeneracy is benign here,
but it would stop being benign the moment `flow_decode_steps > 1`, and the score
must not be read as evidence that the layer is robust.

## 4. Group probes, end to end at temperature 0

Each row ternarizes **only** that group, everything else untouched, GPTQ,
5 utterances, FP-vs-FP floor exactly 1.0000.

| ternarized | layers | params | mel_corr | min | EOS failures |
|---|---|---|---|---|---|
| `flow_net.time_embed` | 4 | 0.8M | **1.0000** | 1.0000 | 0/5 |
| `flow_adaln` (all adaLN) | 7 | 5.2M | 0.8046 | 0.5713 | 0/5 |
| **whole Mimi decoder** | 17 | 10.3M | **0.7721** | **0.7549** | **0/5** |
| backbone `attn_qkv` + `ffn_up` | 16 | 47.7M | 0.6341 | 0.2838 | 1/5 |
| whole backbone | 24 | 75.5M | 0.6298 | 0.2221 | 1/5 |

Read against the 0.648 reseed floor from §1, not against 1.0.

**The compute-heavy half is the half that tolerates ternary.** The Mimi decoder
is 75.4% of the per-frame MACs and it ternarizes with a *tight* spread — the
worst utterance (0.7549) is barely below the mean (0.7721), no termination
failures, and the long utterances behave exactly like the short ones. That is
what a non-autoregressive stage looks like: the error is additive noise on a
fixed input, it does not feed back, and it does not accumulate with utterance
length.

**The backbone is the opposite, and it saturates.** Ternarizing 47.7M
parameters and ternarizing 75.5M give the same answer (0.634 vs 0.630). Damage
that does not grow when you nearly double the perturbed weights is not
accumulated arithmetic error; it is the AR loop finding a different trajectory
as soon as it is pushed at all.

## 5. The failure mode is EOS on long utterances, and it binds early

The aggregate mel correlation hid this; the per-utterance table shows it. At
every cumulative point, short / medium / conversational hold up
(0.89–0.93 at 25%) while the 16-second utterance fails to terminate: duration
ratio 1.70 at 10% ternary, 1.68 at 25%, 1.75 at 50%, 2.31 at 60%, with RMS
falling from 0.058 to 0.011 — the model says the sentence and then emits quiet
garbage instead of stopping. Upstream logs it as
`Maximum generation length reached without EOS`.

It is not a threshold artifact of our `collapsed` flag and it is not caused by
the layers the ranking put first: `time_embed` alone is **bit identical** and
`adaLN` alone never fails to terminate. The failure arrives with the first
backbone layer in the set.

`flow_lm.out_eos` is a 1×1024 projection — 1,024 parameters, below the
eligibility floor, so it stays FP in every configuration here. What ternary
damages is not the EOS head, it is the hidden state handed to it.

**Consequence for serving**: a non-terminating request is not a quality
regression, it is a stuck slot. Under the E5 continuous-batching server it
holds a place until `--max-steps`, and the long class is exactly the class the
v2 corpus is built from. Any ternary backbone would have to clear this gate
before anything else, and mel correlation is the wrong instrument to detect it —
duration ratio and EOS step are the right ones.

## 6. Ternary buys traffic, never arithmetic — and that decides where it can pay

**No ISA we target has a sub-byte multiply-accumulate.** NEON SDOT and SMMLA,
AVX-512 VNNI (`VPDPBUSD`) and AMX are all int8-lane. `src/qmat.c:1403` already
shows the shape this takes: `q4_unpack_u8()` widens nibbles to int8 *before* the
vector dot product. A 2-bit weight would do the same. So ternary's MACs per
instruction are **identical to int8's**, and its entire CPU case is weight
traffic and cache residency.

That matters because of where the traffic actually is:

| component | MMAC/frame | % MACs | ternary MB | % bytes |
|---|---|---|---|---|
| `mimi_decoder` | 157.6 | 46.0 | 1.02 | 4.3 |
| `mimi_dec_tr` | 100.7 | 29.4 | 1.61 | 6.7 |
| `backbone` | 75.5 | 22.0 | **19.10** | **79.4** |
| `flow_head` | 8.9 | 2.6 | 2.31 | 9.6 |

The codec's weights are small and re-read hundreds to thousands of times per
frame, so they are cache-resident and their traffic is already near the floor —
and they are **already int8 by default** (`POCKET_QG_DEFAULT_SPEC`), so ternary
would save 7.7 MB of footprint on a region whose arithmetic it cannot speed up.
The backbone's 151 MB at bf16 are read once per step and fit in no cache, which
is the wall `.work/backbone-bandwidth.md` measured at 32–38 GB/s.

**So the bandwidth argument lands on the backbone, and the backbone is the part
that fails quality.** That tension is the result of this item, and it is why the
E10 rejected-list entries — batched int4 GEMM at 0.80–0.97× on three x86 boxes,
ConvTranspose int8 slower than f32 sgemm — are load-bearing here rather than
background colour: both are measurements of the same unpacking cost, on the same
two regions.

## 7. The metric threshold was wrong, and the rendered audio caught it

The `collapsed` flag started at duration ratio outside [0.6, 1.6]. That was a
guess. The FP-vs-FP reseed control at the shipped temperature, on the seven-text
set, gives the real spread: **0.90 to 1.12**. So a 1.49 is not "inside
tolerance", it is four times the natural variation, and the flag was reporting
`ternary_backbone` as 0/7 clean when the three long utterances were running
**+49%, +16% and +51%** over the reference.

Corrected reading of the shipped-temperature A/B (7 utterances, temp 0.3, FP
reseed floor **0.667 short/medium, 0.538 long**):

| config | dur ratio (long) | mel_corr (long) | RMS | verdict |
|---|---|---|---|---|
| reference | — | — | 0.1129 | — |
| `int8_all` | 1.04 / 0.94 / 0.99 | 0.43 / 0.33 / 0.63 | 0.1182 | duration inside FP spread, level correct |
| `int8_backbone` | 0.92 / 0.99 / 1.00 | 0.32 / 0.82 / 0.61 | 0.1146 | same |
| `ternary_codec` | **1.00 / 1.00 / 1.00** | **0.78 / 0.80 / 0.79** | 0.0695 | duration *exact*; **38% quiet** |
| `ternary_backbone` | **1.49 / 1.16 / 1.51** | 0.15 / 0.33 / 0.16 | 0.0993 | **fails** |

`ternary_codec` is the only configuration whose duration ratio is exactly 1.00
on every utterance — tighter than FP-vs-FP itself. That is not luck: the codec
is deterministic given the latents, so the trajectory is bit-for-bit the
reference's and only the rendering differs. It is also why its mel correlations
are so uniform (0.78–0.81 across every length) while everything touching the
backbone fans out with utterance length.

## 8. Why the ternary codec came out 38% quieter — and the fix

Per-layer gain `‖Ŵx‖ / ‖Wx‖` on the calibration sample, over the 18 ternary
codec layers: **geometric mean 0.9766**. Eighteen in series is
`0.9766^18 = 0.654`; measured output ratio was `0.0695 / 0.1129 = 0.616`.

This is not a bug, it is the estimator. A least-squares ternary fit is a
projection: it minimises `‖Wx − Ŵx‖`, and the minimiser is **biased low in norm
by exactly the fraction of variance it cannot explain**. One layer loses 2%.
A deterministic cascade multiplies the bias while averaging away the noise, so
the loss compounds and the error does not.

`gain_correct()` divides the grid by the measured gain — trading a little MSE
for an unbiased norm. Measured cost across the 18 layers: output NMSE rises from
0.0811 to 0.0828 on the worst layer, i.e. **~2% relative**, and is unchanged to
four decimals on half of them. That is the right trade in a feed-forward
cascade. It is **not** obviously right inside the AR loop, where the state is
fed back and an inflated norm could compound the other way; that is measured
separately and not assumed.

**Anyone implementing ternary in the runtime would hit this**, and the natural
misdiagnosis is a packing or scale bug in the kernel. It is neither.

## 9. KOTMS: the rotation is real, and it still loses

Eight calibrated layers, `build/ternary-kotms/sweep.json`, output NMSE:

| layer | naive | aga | **gptq** | kotms |
|---|---|---|---|---|
| backbone `layers.0.linear1` | 0.0720 | 0.0214 | **0.0026** | 0.0058 |
| backbone `layers.0.linear2` | 0.2170 | 0.0511 | **0.0201** | 0.0509 |
| backbone `layers.0.qkv` | 0.0345 | 0.0025 | **0.0004** | 0.0022 |
| `flow_net.res_blocks.0.mlp.0` | 0.1022 | 0.0369 | **0.0050** | 0.0224 |
| `mimi.decoder.model.0.conv` | 0.1043 | 0.0767 | **0.0096** | 0.0579 |
| `dec_tr layers.0.linear1` | 0.0842 | 0.0467 | **0.0143** | 0.0401 |
| `dec_tr layers.0.linear2` | 0.1579 | 0.1385 | **0.0314** | 0.1141 |
| `dec_tr layers.0.qkv` | 0.1658 | 0.0920 | **0.0373** | 0.0733 |

TWLA's tri-modal shaping **works** — it beats AGA on all eight layers, by
1.2–3.7×, so the premise that a rotation can make a weight distribution more
ternary-friendly is not wrong. It is simply dominated: **GPTQ-style sequential
compensation beats it by 2–6× on every layer**.

And the two are not equally priced. GPTQ is a *build-time* transform: it
changes which ternary values get stored and costs the runtime nothing. KOTMS
leaves a rotation `R = R₁ ⊗ R₂` that has to be applied to the **activation on
every call**, forever. Kronecker structure makes that `n₁²+n₂²` instead of `d²`,
which is why TWLA uses it, but it is not free and it is a new kernel of its own.

So the answer to "PT²-style or TWLA-style for this TTS" is **neither**: the part
both papers share — activation-aware grid fitting — is worth 1.1–8.9×, the part
that is actually decisive is GPTQ-style compensation at 1.3–108×, and the parts
that distinguish the two papers (SSR, KOTMS) are second-order and, in KOTMS'
case, come with a permanent runtime cost. Caveat, stated plainly: our KOTMS is
random-direction descent on the TriGMM surrogate, not Adam through the Cayley
map. A better optimiser would narrow the gap. It would have to narrow it by
6× *and* pay for the rotation to change the conclusion.

## 10. What this item did NOT establish

- **One voice, one language, six calibration utterances, seven eval sentences.**
  Enough to rank layers and to kill a bad idea; not enough to promote anything.
- **No intelligibility metric.** There is no ASR in this repo and none was added.
  "The long sentence runs 51% over and the RMS collapses" is a termination
  failure, not a WER measurement, and the difference matters if anyone wants to
  argue that a degraded-but-terminating configuration is acceptable.
- **The int8 backbone result is a screen, not a promotion.** 0/7 terminations at
  both temperatures, level correct, duration inside the FP spread — on seven
  sentences, in Python, on a Mac, against the upstream implementation and not
  against `engine_pocket.c`. The repo's own rule applies: a screen may not
  promote, only a 30-minute soak may.
- **Nothing here was measured on the C runtime.** Every number is a property of
  the *checkpoint*, which is the right scope for a feasibility question and the
  wrong scope for any performance claim.
