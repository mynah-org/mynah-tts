# The dtype-conversion hunt — and what it actually found

Prompted by the author's doctrine: *"far profiling e rimuovere tutti i conv
datatypes inutili, e levare scalar blas ecc, misurando man mano i colli di
bottiglia."* Measured 2026-09-12 on M1, PocketTTS EN/IT packs.

**Headline: the first half of that advice found nothing here, and the second half
found two production-blocking bugs.** On ARM/macOS no hot-path conversion is worth
removing — every one measures under 3% against a 20% bar. The wall is **weight
bytes**, not conversions. What paid was looking at what the *production* build
does, and the answer was that it did not build.

Related: [`no-blas.md`](no-blas.md), [`linux-production.md`](linux-production.md),
[`engineering-method.md`](engineering-method.md) §5.

---

## 1. The two measurements that settle the conversion question

**f16 is 1.96× f32 end to end, against a weight-byte ratio of 2.00×.** The gap is
≈0, which means the f16→f32 convert inside the loop **costs nothing** — it issues
in the shadow of the loads it feeds. So there is no win available from `fp16fml`
or `bfmmla` on the convert side. Measured, not estimated.

**Sampling profile on f16 (M1, 4 threads):** `matvec_f16` is **5749 of ~7100
non-idle samples = 81%**. Everything else: `conv1d_apply` 213 (3%), `axpy` 256,
`seanet_decode` 148, `dot_f32` 171, BLAS ~100, and **every memcpy in the process
together 29 (0.4%)**.

That 0.4% is the number that closes most of the candidate list below before it
starts.

Reference cost map (f32, level 2): `codec.transformer` 877 ms (41.8%),
`step.backbone` 681 (32.5%), `prep.decoder_prefill` 310 (14.8%),
`codec.conv_stack` 160 (7.6%), flow 68 (3.2%). The real bottleneck is that
`codec.transformer` makes **16 full passes over the same 6.3 MB of weights per
frame** — that is E3-5b, not a conversion.

---

## 2. Every conversion found, and why each stays

| # | where | what | size | loop or load | verdict |
|---|---|---|---|---|---|
| 1 | `weights.c:122` | bf16→f32 | 109.5M elements → **438 MB resident** | load, cached | stays: conv/norm/bias/embedding still read f32. RTF cost ≈0 |
| 2 | `qmat.c cache_insert` | f32→f16 | ~82M elements, **+179 MB RSS** | load | stays: it *is* `bf16→f32→f16`, two conversions — but the middle is a permanent copy other consumers still need, not a scratch buffer |
| 3 | `qmat.c matvec_f16` | f16→f32 in loop | 2 converts per 8 weights, ~82M/step | **loop** | stays: measured free (§1) |
| 4-5 | `qmat.c` | f32→int8, int32→f32 dequant | ~158K / ~203K per frame | **loop** | stays: int8 only, which already fails oracle parity; never runs under f16 |
| 6 | `seanet.c:287-293` | tap gather of the **weights** (layout, not dtype) | **1.96M floats = 7.9 MB per frame**, redone every frame on constant weights | **loop** | stays: biggest candidate on paper, but all of `conv1d_apply` is 3% of wall. Ceiling ~3%, under the noise |
| 7 | `seanet.c:256` | `previous`→`window` copy, **provably redundant** | 4096 floats/frame | **loop** | stays: ~0.05% |
| 8 | `seanet.c:257` | input→`window` copy | 340K floats/frame | **loop** | stays: removable only with a `(dst, pitch)` contract across the whole op chain; in-place ELU and residual add block it |
| 9 | `engine_pocket.c:2208,2236` | two 512×16 transposes | 64 KiB/frame | **loop** | stays: see the 0.4% memcpy total |
| 10 | `transformer_ar.c` | pure memcpy when `out_norm == NULL` | 32 KiB/frame | **loop** | stays |
| 11 | `engine_pocket.c:550-562` | Box-Muller RNG in **double** | 16 pairs/frame, sqrt/log/sin/cos | **loop** | stays: ~832 transcendentals per request, not measurable |
| 12 | `qmat.c:1355` | `cache_lookup`: mutex + O(N) `strcmp` | ~1700 per request | **loop** | stays **for single-stream**: the `__psynch_mutexwait` in the profile traces to the thread pool's condvar, not this mutex. **Flagged to the batching lane** — under concurrency it becomes contended |

**The useful negative:** the runtime had already moved every conversion to load
time. There was nothing left in the hot loop to remove. Recording that so nobody
re-runs this hunt.

---

## 3. The silent fallbacks — two were blocking

**A. `MYNAH_QUANT=f16` was a no-op on x86.** `MYNAH_QMAT_F16` was gated on
`__aarch64__ && __ARM_NEON`. Compiling `qmat.c` for x86 emitted **zero** f16
symbols; `mynah_qmat_cache_new()` silently rewrote F16→F32 and the run proceeded
at f32 speed. **The 2× we measured on ARM did not exist on the production
target.** → fixed (`3892ba6`): F16C/AVX2 kernel with runtime CPUID/XGETBV, plus a
scalar half kernel so a host without F16C still gets half the weight bytes rather
than a silent retreat.

**B. No Linux build linked, at HEAD.** `mynah_conv1d_sgemm_enabled()` was trapped
inside `#if defined(MYNAH_USE_ACCELERATE)` (opened `conv1d.c:45`, closed `:99`)
while `probe_sgemm_conv()` and `codec_nanocodec.c` call it unconditionally — its
own `#else` branch was never compiled. Verified independently: compiling
`conv1d.c` without Accelerate emits **0** definitions at HEAD and **1** after the
fix. Both `BLAS=openblas` and `BLAS=scalar` failed to link. → fixed (`3892ba6`).

**The break is ours.** `conv1d.c` was created by the E1 step-2 split (`9ef135e`)
**this session**, and survived because every build we run is an Accelerate build.
This is exactly the "link breaks invisible on modern hardware" trap recorded in
[`linux-production.md`](linux-production.md), and it materialised in our own tree
within hours of writing it down. **E4-13 (CI link-only job) is no longer
theoretical.**

**C. `--self-test` fails on x86 at HEAD**: `qmat u8 level=1 not bit-identical at
row 3 (k=200)`. Pre-existing, reproduced on the HEAD binary, **not touched** — it
belongs to the VNNI commit, and it means that work was never self-tested on x86.
Open.

**D. `seanet.c` has 12 fallbacks and zero rows in `--dispatch-map`.** The worst:
with `BLAS=scalar` the **entire codec conv stack** drops to the hand-written
scalar loops — the path that measured 8570 ms before the GEMM work, **36×
slower** — and nothing says so. `MYNAH_SEANET_BLAS` appears only inside
`seanet.c`; `dispatch.c` contains zero occurrences of "seanet". Open (E4-19).

**E.** The depthwise `upsample` (`groups=512`) always takes
`convtr_scatter_scalar`, silently, every frame.
**F.** `conv1d.c:487,530` ignore the return of `mynah_graph_sgemm`: with a NULL
backend the output stays bias-only, with no error.
**G.** macOS uses BNNS for conv1d, Linux uses sgemm — different paths between dev
and prod. At least this one is visible (`codec.sgemm_conv`).
**H.** `seanet.c:53-55` narrows `size_t→int` for every sgemm with **no guard**, 6
narrowings per call. `conv1d.c` has the guard; `seanet.c` does not. Correctness,
not performance.

---

## 4. The memory finding, which the server will care about

RSS measured: f32 **626 MB**, f16 **805 MB**, int8 **724 MB**.

The f32 conversion cache stays resident in full even when every hot projection is
served from f16. About 82M of the 109.5M parameters are linear weights that could
go **bf16→f16 directly**, saving ~328 MB per pack. It is memory, not RTF, so it
was not done — but **for a prefork server with N resident languages it is the
dominant term**, and it needs `qmat` to take a bf16-aware path instead of an
already-converted `float*`.

Note the interaction with prefork: weights are quantised **before** the fork and
shared copy-on-write, so 438 MB of dead f32 is paid once for the box, not per
worker — unless a worker touches it. Worth measuring PSS, not RSS, before sizing
this.

---

## 5. What was changed

`3892ba6`: the x86 F16C/AVX2 + scalar half kernels, `uint16_t` weight storage
(`__fp16` is ARM-only, `_Float16` on x86 needs GCC ≥ 12; on aarch64 it is a cast
to identical storage, so ARM is unchanged bit for bit), a scalar conversion
reference with a bit-identity self-test over subnormals / the 2^-25 flush point /
the 65504-65520 boundary / tie-to-even, `quant.f16` promoted from boolean to a
named kernel with its reason, and the `conv1d.c` link fix.

The self-test **earned its place immediately**: it caught a missing overflow
branch in the reference itself. The F16C hardware was right; the reference was
wrong.

**Gates:** WAVs byte-identical before/after on ARM for f32, f16 and int8 across EN
and IT — the change is an exact no-op there, so the existing oracle numbers hold
by construction. Goldens 8/8, batch parity PASS, `--self-test` PASS on ARM and
under x86 emulation with both the forced F16C and the scalar kernel. Cross-ISA
x86-f16 vs x86-f32 correlation **0.99999961**, against an existing f32-vs-f32
cross-ISA noise floor of 0.99999956 — same sample count, no extra frame. UBSan,
ASan (79 checks) and leaks clean.

**Not claimed: x86 RTF.** The emulation layer does not expose F16C through CPUID,
so the scalar kernel ran there. It must be measured on the EPYC box before any
number is written down.

**RTF on ARM is unchanged**, as it must be for byte-identical output. It was
measured anyway across 7 interleaved runs and the spread was 24-48% because the
machine was loaded with other builds — which is itself the lesson from
[`engineering-method.md`](engineering-method.md): paired, interleaved arms, or no
number. The clean earlier baseline (best of 5) stands: f32 0.527 / f16 0.269 /
int8 0.225.
