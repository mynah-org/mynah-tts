# Ternary (W1.58) feasibility for PocketTTS

**Question**: does the trained PocketTTS checkpoint tolerate post-training
ternary weights well enough to justify building a ternary GEMV/GEMM backend in
C?

**This document is the cheap half of that question.** No kernel was written. The
measurements below come from `tools/ternary_feasibility.py`, which fake-quantizes
weights in Python, replays real calibration activations captured from the
reference implementation, and generates real audio. The expensive half — a new
qtype through `src/qmat.c`, 2-bit packing, SDOT/SMMLA and VNNI accumulation, a
packer, a cache, a dispatch row and self-tests on both ISAs — is deliberately
not started until this page says whether it is worth it.

Work item: [`PLAN.md` E11](../PLAN.md) · detail: [`.work/ternary-feasibility.md`](../.work/ternary-feasibility.md)

## Why this question is worth asking here specifically

`.work/backbone-bandwidth.md` measured `step.backbone` at **48.7% of
`request.total`**, and found that **eight cores buy 15% of it**. One AR step
reads every backbone weight exactly once — 75.5M weights, 151 MB at two bytes
each — and at 4.2–4.8 ms per step that is 32–38 GB/s that does not move with
thread count. Arithmetic intensity is one MAC per two bytes. That note's
conclusion was blunt:

> There are only two levers on a bandwidth wall, and neither is a kernel.
> **Fewer weight bytes** […] It is blocked on quality and the block is real […]
> This is a quality problem wearing a performance problem's clothes.

E11 is the measurement of that block. Ternary is not an incremental step down
from bf16; it is **8.6× fewer weight bytes** on the region that owns half the
wall clock. If the quality holds, it is the largest lever left on the backbone.
If it does not, that is worth knowing before a kernel exists.

Two prior results from `PLAN.md` E10's rejected list set the bar, and both cut
against ternary:

- **batched int4 GEMM was 0.80–0.97× on three x86 boxes.** Sub-byte weights have
  already lost once in this repo, at batch, because unpacking cost more than the
  bandwidth it saved. Ternary is sub-byte.
- **ConvTranspose int8 was slower than f32 sgemm.** The codec's transposed
  convolutions resisted even int8, and they are a large share of the codec's
  arithmetic.

So "the quality survives" is necessary and nowhere near sufficient. The
performance argument has to be made separately, per region, and this page is
careful to keep the two apart.

## What was implemented, and how faithfully

Three papers were read before the experiment was designed. What was taken and
what was not:

| Method | Source | Fidelity |
|---|---|---|
| `itf` | PT²-LLM [2510.03267](https://arxiv.org/abs/2510.03267) | **Exact.** Alternating closed-form per-row (α, μ) against flexible rounding of T, minimising ‖W − (diag(α)T + μ1ᵀ)‖²_F, to convergence or 10 iterations. |
| `aga` | PT²-LLM | **Exact for the estimator, without SSR.** Refits (α, μ) under the calibration metric `S = XᵀX` with T frozen — the per-row 2×2 system. Structural-Similarity Reordering is not implemented. |
| `e2m` | TWLA [2606.13054](https://arxiv.org/abs/2606.13054) | **Adapted.** E2M-ATQ stage 1 (Euclidean warm start, μ initialised to the per-row mean, residual-mean correction inside the loop, 15 iterations) and stage 2 (manifold relocation). |
| `gptq` | TWLA / Qwen3-4B [2609.01962](https://arxiv.org/abs/2609.01962) | **Adapted.** AGA grid held fixed, columns quantized left to right with the residual pushed through the inverse Hessian. Column order is `diag(H)` descending (GPTQ act-order), standing in for PT²'s SSR. |
| `kotms` | TWLA | **Partial, and labelled as such.** `R = R₁ ⊗ R₂` fitted by random-direction descent on the tri-modal Gaussian-mixture surrogate, not Adam through the Cayley map. The rotation is folded as `Ŵ = Q(WR)Rᵀ`, which is exact for evaluation but hides a runtime cost — see §F. |
| `int8`, `int4` | `src/qmat.c` | **Exact.** Per-row symmetric absmax, the quantizer already in the binary, measured through the same dequantize and the same metrics so the comparison is between quantizers and not between code paths. |

**Deliberately out of scope**: W1.58A4 (low-bit activations), ILA-AMP bit
allocation, and any change to production inference behaviour.

**A finding that came free with the reading**: PT²'s AGA and TWLA's E2M-ATQ
stage 2 are the *same estimator* — both freeze T and solve a per-row 2×2 system
in the metric `S = XᵀX`. The papers differ in notation and in how they reach T,
not in the relocation. Both are implemented anyway, so the gap between them
measures the warm start and nothing else. Anyone budgeting to implement both
should know they are buying one thing.

**A second finding, also free**: the Qwen3-4B paper does not claim a speedup.
Its Triton ternary kernel is **4.6× slower than FP16**. Compression is not
acceleration.

## How the measurement is set up

- **Calibration** is captured from the upstream `pocket-tts` model with forward
  pre-hooks, six English utterances, voice `alba`, seed 1234. Per module we
  accumulate the **exact** Gram matrix `S = XᵀX` over every activation row the
  layer sees, plus a 512-row reservoir sample. Two structures because there are
  two consumers: GPTQ's Hessian inverse is meaningless on a rank-512 `S` of
  dimension 3584, and the output-error metric needs actual rows.
- **The canonical form** every quantizer sees is a 2-D matrix `(rows, contract)`
  where `contract` is the axis the activation is summed over — `in` for a
  linear, `in·k` for a conv, and `in` for a transposed conv whose rows are then
  `out·k`. Getting this axis wrong would make every activation-aware method
  silently wrong, so it is one function with one comment.
- **Nothing is ternarized by default.** A tensor that does not match the
  taxonomy lands in `other` and stays FP. Norms, biases, `layer_scale`, the
  time-embedding frequencies and the depthwise upsample are all excluded with a
  stated reason.
- **The pack is never written.** `Pack` is read-only; end-to-end runs
  monkey-patch a throwaway in-memory model.

The calibration run also produced an independent check on the taxonomy: 18
encoder modules were **never called** during synthesis, exactly the set the
census had marked `streaming: false`. The Mimi encoder is voice-cloning
machinery; it is 9.7M parameters that the streaming loop never reads.


## The answer, up front

**No-go on a ternary backbone. A conditional, small yes on the codec. And the
measurement found a better lever than the one it went looking for.**

| | verdict | why |
|---|---|---|
| ternary **backbone** (75.5M, 79.4% of ternary bytes) | **no** | long utterances over-run by +49%/+51% at the shipped temperature and fail to terminate outright at temperature 0 |
| ternary **Mimi decoder** (10.3M, 75.4% of MACs) | **yes, but it buys little** | clean — duration ratio exactly 1.00 on every utterance — but the region is already int8, and ternary cannot beat int8 on arithmetic |
| **int8 everywhere** | **the actual win** | 0/7 termination failures, level correct, duration inside the FP-vs-FP spread, and **no new kernel is needed** |

The decisive fact is architectural, not statistical:

> **No CPU ISA we target has a sub-byte multiply-accumulate.** NEON `SDOT` and
> `SMMLA`, AVX-512 VNNI `VPDPBUSD` and AMX are all int8-lane. `src/qmat.c:1403`
> already shows the shape: `q4_unpack_u8()` widens nibbles to int8 *before* the
> vector dot product, and a 2-bit weight would do the same. **Ternary's MACs per
> instruction are identical to int8's.** Its entire CPU case is weight traffic.

And the traffic is in the wrong place:

| component | % MACs/frame | % of ternary bytes |
|---|---|---|
| `mimi_decoder` + `mimi_dec_tr` | **75.4** | 11.0 |
| `backbone` | 22.0 | **79.4** |

Ternary's only lever is bytes; 79.4% of the bytes are in the backbone; the
backbone is the part that fails. The region that passes quality holds 11% of the
bytes, is already cache-resident, and is already int8 — so ternary there would
save ~7.7 MB of footprint for **zero** arithmetic gain, against a prior measured
0.80–0.97× for exactly this unpacking cost (E10's rejected list: batched int4
GEMM on three x86 boxes; ConvTranspose int8 slower than f32 sgemm).

### What to do instead

`.work/backbone-bandwidth.md` measured the backbone at 32–38 GB/s, 48.7% of
`request.total`, with eight cores buying 15%. Its conclusion was that the only
lever is fewer weight bytes, and that the lever was "blocked on quality". This
item measured that block:

```
backbone weight traffic, per AR step
    bf16 (shipped today)   151.0 MB
    int8                    75.7 MB     0/7 termination failures, level correct
    ternary                 19.1 MB     3/7 fail at temp 0, +50% duration at temp 0.3
```

**int8 halves the traffic on the region that owns half the wall clock, the
kernels already exist, and it passed every end-to-end gate here.** That is a
bigger and far cheaper win than ternary, and it is a direct challenge to
`POCKET_QG_DEFAULT_SPEC`'s current `backbone:bf16` pin. It is **not** a
promotion — a screen may not promote, only a soak may — but it is now a
measured candidate rather than a suspicion, and it is the recommended next
item.

<!-- BEGIN GENERATED TABLES -->
<!-- generated by tools/ternary_feasibility.py report on 2026-09-20; do not hand-edit the tables -->

## A. Parameter census

`kyutai/pocket-tts` rev `492522650173a0653b7575cdc25ae09810e5d741`, 109,502,146 parameters in 214 tensors, bfloat16 on disk.

| component | tensors | params | % of model | ternary-eligible | in streaming path |
|---|---|---|---|---|---|
| backbone | 48 | 75,522,048 | 69.0% | 75,497,472 | yes |
| flow_head | 68 | 9,759,008 | 8.9% | 9,732,096 | yes |
| mimi_dec_tr | 20 | 6,297,600 | 5.8% | 6,291,456 | yes |
| mimi_enc_tr | 20 | 6,297,600 | 5.8% | 6,291,456 | no |
| conditioner | 1 | 4,097,024 | 3.7% | 4,097,024 | no |
| mimi_decoder | 24 | 4,007,713 | 3.7% | 3,987,456 | yes |
| mimi_encoder | 23 | 3,451,424 | 3.2% | 3,446,784 | no |
| backbone_io | 10 | 69,729 | 0.1% | 65,536 | yes |

## B. Per-layer sensitivity

| group | tensors | params | int8 NMSE | int4 NMSE | naive NMSE | itf NMSE | aga NMSE | e2m NMSE | gptq NMSE | gptq-gc NMSE |
|---|---|---|---|---|---|---|---|---|---|---|
| backbone/ffn_up | 6 | 25,165,824 | 0.0000 | 0.0017 | 0.0492 | 0.0486 | 0.0151 | 0.0151 | 0.0024 | 0.0024 |
| backbone/ffn_down | 6 | 25,165,824 | 0.0004 | 0.1222 | 0.4030 | 0.2897 | 0.1673 | 0.1673 | 0.0120 | 0.0121 |
| backbone/attn_qkv | 6 | 18,874,368 | 0.0000 | 0.0016 | 0.0594 | 0.0567 | 0.0132 | 0.0132 | 0.0028 | 0.0028 |
| backbone/attn_out | 6 | 6,291,456 | 0.0001 | 0.0218 | 0.1772 | 0.1757 | 0.1421 | 0.1424 | 0.0088 | 0.0088 |
| flow_head/flow_adaln | 7 | 5,242,880 | 0.0000 | 0.0016 | 0.0203 | 0.0084 | 0.0023 | 0.0023 | 0.0002 | 0.0002 |
| mimi_enc_tr/ffn | 4 | 4,194,304 | 0.0001 | 0.0388 | 0.2225 | 0.2209 | - | - | - | - |
| conditioner/text_embedding | 1 | 4,097,024 | 0.0001 | 0.0209 | 0.1919 | 0.1911 | - | - | - | - |
| flow_head/flow_mlp | 12 | 3,145,728 | 0.0001 | 0.0274 | 0.1785 | 0.1734 | 0.1063 | 0.1062 | 0.0128 | 0.0128 |
| mimi_encoder/conv | 9 | 2,922,496 | 0.0002 | 0.0526 | 0.3055 | 0.2730 | - | - | - | - |
| mimi_dec_tr/ffn_up | 2 | 2,097,152 | 0.0000 | 0.0040 | 0.0740 | 0.0731 | 0.0332 | 0.0332 | 0.0103 | 0.0103 |
| mimi_dec_tr/ffn_down | 2 | 2,097,152 | 0.0001 | 0.0290 | 0.1795 | 0.1758 | 0.1571 | 0.1572 | 0.0383 | 0.0387 |
| mimi_enc_tr/attn | 4 | 2,097,152 | 0.0001 | 0.0216 | 0.2041 | 0.2011 | - | - | - | - |
| mimi_decoder/conv | 6 | 2,004,992 | 0.0002 | 0.0391 | 0.2906 | 0.2761 | 0.1595 | 0.1561 | 0.0515 | 0.0527 |
| mimi_decoder/convtr | 3 | 1,966,080 | 0.0002 | 0.0339 | 0.2359 | 0.1641 | 0.1137 | 0.1142 | 0.0336 | 0.0341 |
| mimi_dec_tr/attn_qkv | 2 | 1,572,864 | 0.0000 | 0.0119 | 0.2114 | 0.1717 | 0.1155 | 0.1157 | 0.0359 | 0.0362 |
| flow_head/flow_time_embed | 4 | 786,432 | 0.0010 | 0.2046 | 0.6448 | 0.0519 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| flow_head/flow_cond | 1 | 524,288 | 0.0000 | 0.0098 | 0.1106 | 0.1087 | 0.0786 | 0.0784 | 0.0089 | 0.0089 |
| mimi_dec_tr/attn_out | 2 | 524,288 | 0.0000 | 0.0155 | 0.1698 | 0.1643 | 0.1180 | 0.1171 | 0.0272 | 0.0273 |
| mimi_encoder/downsample | 1 | 524,288 | 0.0023 | 0.2654 | 0.4862 | 0.4478 | - | - | - | - |
| backbone_io/latent_in | 1 | 32,768 | 0.0000 | 0.0154 | 0.2380 | 0.2130 | 0.2004 | 0.1974 | 0.1617 | 0.1688 |
| backbone_io/speaker | 1 | 32,768 | 0.0000 | 0.0140 | 0.2265 | 0.2035 | - | - | - | - |
| flow_head/flow_out | 1 | 16,384 | 0.0000 | 0.0068 | 0.1663 | 0.1167 | 0.0601 | 0.0598 | 0.0090 | 0.0091 |
| flow_head/flow_in | 1 | 16,384 | 0.0001 | 0.0159 | 0.2355 | 0.1995 | 0.1965 | 0.1970 | 0.1875 | 0.1972 |
| mimi_decoder/latent_proj | 1 | 16,384 | 0.0001 | 0.0192 | 0.2621 | 0.2039 | 0.1883 | 0.1850 | 0.1580 | 0.1648 |

## C. Cumulative ternarization — gptq

| budget | layers | ternary params | % of model | eff. bits/weight | ckpt MB | mel corr mean | mel corr min | dur ratio | collapsed |
|---|---|---|---|---|---|---|---|---|---|
| 10% | 12 | 9.2M | 8.4% | 14.83 | 203.0 | 0.6791 | 0.2879 | 1.17 | 1 |
| 25% | 17 | 23.9M | 21.8% | 12.96 | 177.4 | 0.6923 | 0.1126 | 1.11 | 1 |
| 40% | 20 | 35.4M | 32.3% | 11.49 | 157.2 | 0.5728 | 0.2032 | 1.03 | 0 |
| 50% | 24 | 45.1M | 41.2% | 10.25 | 140.3 | 0.5328 | 0.1592 | 1.22 | 1 |
| 60% | 32 | 53.7M | 49.1% | 9.15 | 125.2 | 0.5032 | 0.1772 | 1.46 | 3 |
| 70% | 42 | 66.4M | 60.6% | 7.53 | 103.1 | 0.3613 | -0.0088 | 1.49 | 3 |
| 80% | 46 | 73.8M | 67.4% | 6.58 | 90.1 | 0.4365 | 0.1151 | 1.02 | 0 |
| 90% | 50 | 84.3M | 77.0% | 5.24 | 71.8 | 0.4131 | 0.1608 | 1.56 | 3 |
| 100% | 69 | 95.5M | 87.3% | 3.81 | 52.2 | 0.3744 | 0.2902 | 2.52 | 7 |

## C. Cumulative ternarization — int8

| budget | layers | ternary params | % of model | eff. bits/weight | ckpt MB | mel corr mean | mel corr min | dur ratio | collapsed |
|---|---|---|---|---|---|---|---|---|---|
| 100% | 24 | 75.5M | 68.9% | 10.49 | 143.6 | 0.7971 | 0.5200 | 0.99 | 0 |

## D. Kernel shapes that would dominate a C backend

| M x K | op | instances | calls/frame | MMAC/frame | % MACs | % ternary bytes | example |
|---|---|---|---|---|---|---|---|
| 2048x512 | gemv_tall | 2 | 32.0 | 33.55 | 9.8% | 2.2% | mimi.decoder_transformer.transformer.layers.0.linear1.weight |
| 512x2048 | gemv_tall | 2 | 32.0 | 33.55 | 9.8% | 2.2% | mimi.decoder_transformer.transformer.layers.0.linear2.weight |
| 1280x256 | conv_im2col | 1 | 96.0 | 31.46 | 9.2% | 0.4% | mimi.decoder.model.5.convtr.weight |
| 512x128 | conv_im2col | 1 | 480.0 | 31.46 | 9.2% | 0.1% | mimi.decoder.model.8.convtr.weight |
| 1536x512 | gemv_tall | 8 | 38.0 | 29.88 | 8.7% | 6.7% | mimi.decoder_transformer.transformer.layers.0.self_attn.in_proj.weight |
| 512x3584 | conv_im2col | 1 | 16.0 | 29.36 | 8.6% | 1.9% | mimi.decoder.model.0.conv.weight |
| 3072x512 | conv_im2col | 1 | 16.0 | 25.17 | 7.3% | 1.7% | mimi.decoder.model.2.convtr.weight |
| 4096x1024 | gemv_tall | 6 | 6.0 | 25.17 | 7.3% | 26.6% | flow_lm.transformer.layers.0.linear1.weight |
| 1024x4096 | gemv_tall | 6 | 6.0 | 25.17 | 7.3% | 26.3% | flow_lm.transformer.layers.0.linear2.weight |
| 3072x1024 | gemv_tall | 6 | 6.0 | 18.87 | 5.5% | 19.9% | flow_lm.transformer.layers.0.self_attn.in_proj.weight |
<!-- END GENERATED TABLES -->

## Reproduce

```bash
python3 tools/ternary_feasibility.py census
uv run --with pocket-tts --with numpy --with scipy \
    python3 tools/ternary_feasibility.py calib --examples 6
python3 tools/ternary_feasibility.py sweep --methods int8,int4,naive,itf,aga,e2m,gptq
python3 tools/ternary_feasibility.py layout --method gptq
uv run --with pocket-tts --with numpy --with scipy \
    python3 tools/ternary_feasibility.py cumulative --method gptq
python3 tools/ternary_feasibility.py shapes
python3 tools/ternary_feasibility.py report
```

## The ten questions, answered

**1. How many parameters are genuine ternary candidates?**
Structurally, 109.4M of 109.5M (99.9%) are linear/conv tensors with a
contraction ≥ 32 and ≥ 4096 parameters. By *measured tolerance*, **10.3M
(9.4%)** — the Mimi decoder. The backbone's 75.5M are structurally perfect
candidates and fail the audio gate. Structure and tolerance are not the same
question and quoting the first as the second is the trap this whole page exists
to avoid.

**2. Least-sensitive modules** (GPTQ, output NMSE on real activations):
`flow_net.time_embed` 0.0000 (degenerate — see the caveat), `flow_adaln`
0.0002, backbone `ffn_up` 0.0024, backbone `attn_qkv` 0.0028, the large Mimi
decoder convs 0.009–0.011.

**3. Most-sensitive modules**, and they are all tiny: `flow_net.input_proj`
0.1875 (16K params), `flow_lm.input_linear` 0.1617 (32K),
`mimi.quantizer.output_proj` 0.1580 (16K), the decoder's 1×1 residual convs
0.087–0.128 (8–32K), `mimi.decoder.model.2.convtr` 0.0811 (1.6M).
**The protected set costs ~100K parameters, under 0.1% of the model.** These
three latent-boundary projections also gain almost nothing from any method
(1.3–1.7× over naive, against 20–108× elsewhere): they are irreducibly hard,
not badly fitted.

**4. How much does activation-awareness buy over naive ternary?**
AGA 1.1–8.9×; GPTQ-style compensation **1.3–108×**, median ≈ 13×. Per group:
backbone `ffn_down` 0.4030 → 0.0120 (**33.5×**), `flow_adaln` (**108×**),
backbone `attn_qkv` (**21.6×**). Ternary without error compensation is not
competitive with anything.

**5. PT²-style or TWLA-style?**
**Neither, because on the part that matters they are the same algorithm.** AGA
and E2M-ATQ stage 2 agree to within 0.3% on every group — both freeze T and
solve the identical per-row 2×2 system under `S = XᵀX`.

TWLA's distinguishing piece, KOTMS, was implemented and measured on eight
layers. The tri-modal rotation **works** — it beats AGA on all eight, by
1.2–3.7× — and it still loses to GPTQ by **2–6× on every one**. It is also the
only method here that leaves a permanent runtime cost: `R = R₁ ⊗ R₂` must be
applied to the activation on every call, forever, whereas GPTQ is a build-time
transform that costs inference nothing.

So: the shared part (activation-aware grid fitting) is worth 1.1–8.9×, the
decisive part (GPTQ-style compensation, which both papers adopt) is worth
1.3–108×, and the parts that actually distinguish the two papers are
second-order. Do not budget to implement both, and do not budget for the
rotation.

**6. How far before speech degrades?**
The whole codec, with no degradation beyond the FP-vs-FP reseed floor. The
backbone, not at all: the first backbone layer added to the set is enough to
start long utterances over-running. Short/medium/conversational tolerate far
more than long ones — the damage is a function of utterance length, because it
is AR feedback and not accumulated arithmetic.

**7. Best mixed layout?**
`ternary codec + int8 everything else`. Every layout the tool generated that
reached an interesting bit budget got there by ternarizing the backbone, and
every one of those failed.

**8. Real effective bits/weight**, with scales and metadata counted:

```
codec ternary   10,278,912 params ->   2.63 MB   2.05 bits/weight
rest int8       99,223,234 params ->  99.68 MB
TOTAL          109,502,146 params -> 102.32 MB   7.48 effective bits/weight
bf16 today                            219.00 MB   -> 2.14x smaller
```

The nominal "1.58" never appears. Even on the ternary part alone the honest
figure is **2.05 bits/weight** once the 2-bit plane, the per-row scale and the
per-row shift are counted.

**9. Is there enough evidence to justify a C ternary GEMV/GEMM backend?**
**No.** Four independent reasons, any one of which would be enough:
(a) ternary cannot beat int8 on MACs per instruction on any ISA we target, so
its only lever is traffic; (b) 79.4% of the traffic is in the backbone; (c) the
backbone fails the audio gate; (d) the region that passes holds 11% of the
bytes, is cache-resident, and is already int8 — and this repo has already
measured the sub-byte unpacking cost at 0.80–0.97× on three x86 boxes.

**10. If it were a go, the first shapes** — recorded so the analysis is not
repeated, not as a recommendation. From `shapes.json`, by MACs per frame:
`2048×512` and `512×2048` (decoder-transformer FFN, 32 calls/frame, 9.8% of
MACs each), `1536×512` (decoder-transformer QKV, 8 instances, 8.7%),
`512×3584` (`mimi.decoder.model.0.conv` via im2col, 8.6%), `3072×512`
(`model.2.convtr`, 7.3%). All would be int8 GEMV after unpacking, which is
precisely why they are not worth a new kernel.
