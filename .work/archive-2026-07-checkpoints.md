# Archive — `PLAN.md` §17-24 (July-August 2026)

Status: **ARCHIVE** — moved verbatim from `PLAN.md` on 2026-09-12 when the plan
was converted to a board-plus-notes layout. Nothing here was edited.

These are dated session checkpoints, measurements, accepted and rejected ideas
from the Magpie v1 work. They are kept because the **rejected** items matter:
Metal on Apple Silicon, vectorizing Snake, cross-request codec batching and
pool-threading the codec convolutions were all measured and closed. Re-trying
them without reading this is wasted work.

Measurements that are still current live in `docs/performance.md`. Where this
archive and `docs/performance.md` disagree, `docs/performance.md` wins.

---

## 17. End-of-day checkpoint — 2026-07-23

The native Magpie command now writes a longer WAV than the original writer
smoke test: the current sample is about 1.1 seconds at 22050 Hz, with valid
non-silent PCM. It is not intelligible speech yet and must not be treated as a
successful v1 quality result. The previous sub-second WAV was only a 440 Hz
writer test tone, not model output.

Tomorrow's first TODO is to inspect and correct the audio path before adding
more model families or GPU optimization:

1. Run the NeMo/PyTorch oracle and compare token IDs, decoder logits, EOS,
   stacked codebooks, local-transformer output, and NanoCodec output stage by
   stage.
2. Isolate the first divergent stage, with special attention to speaker/text
   conditioning, codebook ordering and EOS trimming, causal codec alignment,
   and the normalized-input contract.
3. Fix the native implementation, regenerate the WAV with fixed seed and
   parameters, and verify that the result is intelligible before calling v1
   complete.

## 18. Checkpoint — 2026-07-24: intelligible English + open work

The native English path now produces intelligible speech ("hello world",
speaker 4, 22050 Hz). Fixes landed this session:

- `embed_audio` read the codes buffer with the growing current length instead
  of the `max_raw_length` row stride, corrupting the decoder's audio history
  for every codebook after the first.
- Incremental decoder KV cache (per-layer self-attention K/V + constant text
  cross-attention K/V + baked-context prefill), replacing the per-step full
  recompute.
- Codec `conv1d` vectorized as accumulated BLAS sgemm taps.
- EOS: `audio_eos_id` is `codebook_size + 1` (2017), not `codebook_size`
  (2016 = BOS). Special tokens are forbidden except AUDIO_EOS; stop fires on
  argmax-or-multinomial in any codebook (matches NeMo). Ids now come from
  `model.json`.

### Performance target (do this next)

CPU release gate is **RTF <= 0.2** and the stretch/product goal is **RTF <=
0.1** on Apple arm64, excluding model load. Only serial A/B measurements are
accepted: one `mynah-tts` process, `MYNAH_THREADS=1`, same open model, two
warm-ups and 5–7 measured runs. Current M1/T8103 results:

| change | old RTF | new RTF | delta | correctness |
|---|---:|---:|---:|---|
| four-row INT8 SDOT | 0.601 | 0.538 | -10.5% | WAV byte-identical |
| four-row Q4 SDOT | 0.637 | 0.595 | -6.6% | WAV byte-identical |
| causal conv im2col | 0.589 | 0.503 | -14.6% | max 1 int16 LSB; corr 0.9999999995 |
| Accelerate/vForce Snake | 0.534 | 0.478 | -10.5% | max 1 int16 LSB; corr 0.9999999998 |
| reused local-step scratch | 0.515 | 0.488 | -5.2% | WAV byte-identical |
| reused local sampler scratch | 0.577 | 0.480 | -16.8% | WAV byte-identical |
| decoder-prefill vForce GELU (greedy) | 0.476 | 0.424 | -10.9% | ranges 0.465–0.486 / 0.417–0.439; 29,696 samples; WAV byte-identical |
| reused codec im2col workspace | 0.457 | 0.417 | -8.8% | ranges 0.438–0.479 / 0.410–0.421; reversed confirmation 0.494→0.417; WAV byte-identical; peak RSS -35 MB |
| Accelerate BNNS causal codec conv | 0.446 | 0.394 | -11.7% | ranges 0.430–0.573 / 0.359–0.420; same 29,696 samples; max 1 int16 LSB; corr 0.999999999744 |
| dead im2col workspace removal (BNNS) | 0.368 | 0.378 | ~0% | footprint fix, not speedup; peak RSS −5 MB; WAV byte-identical |
| decoder scratch reuse + pre-resolve | 0.344 | 0.344 | ~0% | correctness fix (zero alloc/lookup in AR loop); WAV byte-identical |
| NEON axpy attn·V (decoder + local) | 0.344 | 0.327 | -4.9% | ranges 0.340–0.349 / 0.326–0.329; WAV byte-identical; Q·K stays scalar |

INT8/Q4 absolute RTFs are not f32 speedups because quantization changes
generated duration/EOS; only same-mode A/B deltas above are comparable.
Rejected on M1: four-thread qmat row partitioning regressed 0.511→0.554 RTF;
GCD dispatch also lost to the existing pthread path (0.511 vs 0.484). Both
experiments were removed. Applying vForce GELU to the text encoder was also
removed: it changed the greedy token/EOS path and produced an incomparable
0.511 RTF result. The accepted GELU path is restricted to multi-row decoder
prefill; one-row autoregressive steps and the encoder retain scalar `tanhf`.
A one-row vDSP layer-norm path was removed too: its byte-identical 0.433→0.422
result (-2.5%) did not clear the 3% acceptance floor. Later rejected experiments:
reused Snake scratch (no repeatable win and worse median on reversed A/B),
kernel-3 tap GEMMs (-1.6%, below threshold), eight-row INT8 SDOT (+3.9%
regression), 4x16 INT8 weight packing (no median change), and pointer-based
qmat cache lookup (no repeatable win). A 1→4 Accelerate thread sweep improved
only 1.6%, also below the acceptance floor. NEON Q·K dot product was rejected:
4-lane accumulation reordering shifted softmax scores enough to flip greedy
argmax tokens (24,576 vs 29,696 samples); Q·K stays scalar for token-path safety.
Q4 quantization was measured and rejected as a speedup: per-stack it is +5–9%
slower than INT8 because M1 matvecs are compute-bound (weights fit in L2), and
the Q4 nibble-unpacking overhead exceeds the halved memory traffic.  Q4 also
changes EOS (22,528 vs 29,696 samples).  Kernel fusion and batched matmat were
analyzed and rejected: activation ops are only ~13% of decoder step time (max
~1.8% total gain), and the 16 local-transformer streams are sequential (each
depends on the previous stream's sampled token), so no batching is possible.

Metal (P8) was validated on M1: self-test passes, pooled IO buffers and a tiled
kernel are committed.  Per-op dispatch measures 13.6 ms/call × ~3,300 calls =
RTF 40.7.  Even with perfect 1-commit-per-step batching the overhead alone would
give RTF ~0.48, still slower than CPU INT8 (0.32).  Metal is not viable for
768-wide matvecs on M1; it becomes relevant for dots.tts (2B, GPU-first) or
batched inference.

Follow-up Metal tensor A/B on the M1/T8103 used the same f32 Magpie prompt,
speaker, seed, 20-step cap, one process, `MYNAH_THREADS=1`, two warm-ups and
five runs. Apple `MPSMatrixMultiplication` was rejected for every matmul:
GPU-only RTF 4.576 versus 3.912 for the custom kernels. Restricting MPS to
multi-row GEMMs while retaining the row-tiled autoregressive kernel reduced
GPU-only RTF to 3.185 (−18.6%, ranges 3.077–3.271); the MPS result was WAV-byte
identical to the custom GPU-only path.

The direct per-op simdgroup matvec remains useful only as a diagnostic:
commit/wait overhead makes that graph slower than Accelerate. The resident
decoder path also changes EOS/duration through autoregressive floating-point
drift, so it remains opt-in via `MYNAH_METAL_GPU_ATTENTION=1`. The default
Metal path preserves CPU decoder parity, keeps ordinary transformer matmuls on
Accelerate, and uses the resident Metal codec.

The codec bottleneck was the naïve per-output convolution kernel. Replacing it
with resident `im2col` plus cached `MPSMatrixMultiplication` reduced codec time
from about 1.65 s to 0.16 s on the 17-stack validation case. Complete synthesis
improved from median RTF 2.31 to 0.782 (range 0.780–0.810; two warmups, five
serial runs, one process, `MYNAH_THREADS=1`). CPU median was 0.800. Tokens/EOS
and the 32,768-sample duration match; PCM correlation is 1.000000000 and
normalized RMSE is 1.74e-6. The CPU/Metal difference is below the 3% noise
floor, so no CPU speedup is claimed.

### Metal GPU TODO — closed 2026-07-29: Metal is not the lever on Apple Silicon

The resident-graph queue below is **abandoned as a performance effort**, on
measurement rather than on effort remaining. Magpie decode is bound by DRAM
bandwidth, not by compute, and Apple Silicon memory is unified: the GPU draws on
the *same* ceiling as the CPU and can only add dispatch overhead.

Measured on M1 (T8103), AC power, one 6.0 s utterance, one warmup and three
serial runs:

| path | RTF |
| --- | --- |
| CPU f32, 4 threads | **0.542** |
| CPU int8, 4 threads | 0.554 |
| Metal, default threads | 0.625 |
| CPU f32, 1 thread | 0.658 |

Metal is 15% *slower* than the CPU path, and the full device-resident AR graph
(`MYNAH_METAL_GPU_ATTENTION=1`) measured RTF 1.214 against a CPU 0.862 on the
short bench — 42% slower even before its token/EOS parity failure is counted.
Completing it would not have produced a speedup.

The supporting arithmetic: the local transformer streams ~1.0 GB of weights per
stacked frame (16 sequential streams × 2 layers × 28.3 MB, plus one 6.2 MB
output projection per stream), because every stream re-reads the full weight set
from DRAM. That was 8.0 GB in 0.292 s ≈ 27.5 GB/s, against a measured M1 ceiling
of ~59 GB/s across four cores (a 262 MB working set; a smaller set measures
>100 GB/s and only reports cache, not DRAM).

- [~] **P0 — Validate resident local transformer:** dropped. The code remains
  behind `MYNAH_METAL_GPU_ATTENTION=1` as a correctness/diagnostic path only.
- [x] **P0 — Validate resident NanoCodec:** causal convolutions now use
  `im2col` + MPS GEMM with resident buffers. M1 duration, finite/peak/RMS and
  waveform-correlation gates pass; codec time is about 0.16 s versus 1.65 s
  for the rejected naïve Metal kernel.
- [x] **P0 — Implement final device clip:** waveform clipping and non-finite
  replacement now run in a Metal postprocess kernel; real-device parity and
  complete synthesis RTF are still part of the validation gate.

Keep the Metal backend for correctness coverage, for the backend seam CUDA
shares, and for dots.tts, which is GPU-first. Do not re-open it as a Magpie
latency optimization without first showing the bandwidth analysis is wrong.

**Where the bandwidth actually went (accepted 2026-07-29).** Decode projections
are single-row and Accelerate never threads an `M=1` sgemm, so every projection
ran on one core's share of DRAM bandwidth regardless of `MYNAH_THREADS` — a
thread sweep moved the total from 0.602 s to 0.593 s, i.e. not at all. Splitting
the output rows over the pool is bit-exact (each output row is an independent
dot product; no reduction is reordered) and is now the default on
Accelerate/ARM when the pool has more than one worker. `qmat.c` was entirely
single-threaded, which is why int8 — 4× fewer weight bytes, exactly what a
bandwidth-bound loop wants — had been *slower* than f32; it now splits on the
same bit-exact row blocks. The pool defaults to the performance-core count on
Apple Silicon: on M1 (4P+4E) 4 threads gives 0.482 s against 0.507 s for all 8,
because the E-cores add no bandwidth but lengthen every barrier.

Results, short bench (`h|ə|ˈ|l|o|ʊ|<space>|f|ɹ|ʌ|m`, 20 steps, two warmups,
five serial runs), every output byte-identical to the pre-change baseline:

| change | old RTF | new RTF | delta | correctness |
| --- | --- | --- | --- | --- |
| threaded row-split matvec + P-core pool (f32) | 0.862 | 0.692 | −20% | identical md5 at 1/2/4/8 threads |
| threaded qmat row blocks (int8) | 0.831 | 0.650 | −22% | identical md5 at 1/2/4 threads |

The accepted headline number is the **interleaved** A/B, not the table above:
absolute RTF on this machine drifts several percent between batches minutes
apart (the same f32/4-thread configuration measured 0.542 in one batch and 0.579
in another), so old and new must alternate within one batch. Old behaviour is
reachable from the same binary via `MYNAH_CPU_MATVEC=0 MYNAH_THREADS=8`, which
makes the interleave trivial and removes any build difference as a variable.
On the 6.0 s utterance, three alternating repetitions:

| arm | rep1 | rep2 | rep3 | median |
| --- | --- | --- | --- | --- |
| old (Accelerate, 8 threads) | 0.691 | 0.686 | 0.693 | 0.691 |
| new (default) | 0.579 | 0.578 | 0.583 | **0.579** |

−16.2%, byte-identical output (same md5). Note how much tighter the old arm's
spread is than the cross-batch numbers — that spread is the real noise floor,
and it is what a 3% accept/reject rule should be measured against.

Note: the previously recorded "best RTF 0.320 (INT8, M1, 1 thread)" does not
reproduce on this machine — int8 at one thread measures 0.786 on the short bench
and 0.637 on the 6.0 s utterance. Treat 0.320 as stale until someone re-derives
the exact text, step cap and settings behind it.

Next levers, in bandwidth-per-unit-effort order: ~~the fused f32 matvec+argmax
used for the local output projection (`mynah_matvec_argmax_f32`, ~99 MB per
stacked frame) is still single-threaded and needs a per-block argmax reduction
that keeps first-on-ties~~ — done 2026-07-30, −5.8% RTF on f32 greedy,
byte-identical, see docs/performance.md; then prep (0.131 s) and codec
(0.085 s), which are fixed costs now large enough to dominate a short
utterance.

### Quantized decode was barely wired up (fixed 2026-07-29)

The single most valuable finding of this session was structural, not numerical:
**`qmat` had exactly one call site in `graph.c`** -- the local output
projection. The local transformer's qkv/o/ffn matmuls and every decoder
projection called `mynah_backend_matmul` with raw resolved pointers, so
`MYNAH_QUANT=int8` was quantizing roughly a tenth of the decode weight traffic.
That is why INT8 had measured *slower* than f32 and was written off. Routing
both through `mynah_qmat_linear_resolved` (the names were already stored beside
the pointers in `decoder_layer_resolved`; `local_cache` needed them added):

| mode | before routing | local routed | + decoder routed |
| --- | --- | --- | --- |
| f32 | 0.545 | 0.533 | 0.532 (md5 unchanged) |
| f16 | 0.533 | 0.416 | **0.376** |
| int8 | 0.517 | 0.346 | **0.279** |

f32 is bit-exact throughout: with the cache disabled the call falls straight
back to `mynah_backend_matmul`, verified by an unchanged WAV md5 at every step.
Prefill stays exact too -- above `QMAT_SMALL_COUNT` rows the quantized path is
not taken.

**F16 added as a third quantization type.** Weights as IEEE half, activation and
accumulation in f32, no scales, four rows in flight. On Magpie weights the mean
relative error is 1.8e-4 against 1.7e-2 for INT8 (~96x tighter) and the largest
weight is ~6 against the 65504 limit. Measured 2.4-2.8x against Accelerate
sgemv on a working set too large to cache.

**F16 does not preserve the greedy trajectory.** This was the hypothesis worth
testing and it failed: even at 1.8e-4 it flips near-tie argmaxes, and over ~65
frames EOS moves (132096 → 137216 samples). So F16 is not a transparent f32
replacement; it is the accurate end of the existing quantization opt-in, and
belongs in the same category as INT8/INT4 rather than in the bit-exact category
with the threading work above.

Judge quantized output with alignment-insensitive measures. Once the trajectory
diverges the audio is a different valid realization, so sample-wise correlation
collapses (0.16 for f16) and means nothing. Long-term average spectrum
correlates at 0.979 (f16) and 0.973 (int8), with matching RMS, peak and
plausible duration -- that is the gate to use.

### Snake: the analysis below was of dead code (corrected 2026-07-29)

The Snake investigation recorded next was wrong in an instructive way, and the
error is worth keeping. `half_snake` in graph.c has three branches: a device
branch guarded by `backend != NULL`, an Accelerate/vDSP branch, and a scalar
fallback. **The CPU codec always has a backend object**, so the device branch is
always taken and lands in `mynah_backend_snake_dev`'s CPU fallback in backend.c
-- a scalar `sinf` loop over channels with no threading. The vDSP code in
graph.c never runs on any normal CPU synthesis.

Every symptom followed from that and none of it made sense until the branch was
found: `MYNAH_SNAKE_SCALAR=1` changed nothing (0.313 s both ways) because
neither graph.c branch was reachable; a fused tiled rewrite of the vDSP path
measured 1.09-1.45x in a microbenchmark and 0% in the application; and
parallelizing the vDSP path over channels moved the codec by 1 ms. The tell was
that last one -- a change that is clearly correct and clearly does nothing means
the code is not running.

Threading the *real* fallback in backend.c is bit-exact (each channel is an
independent row, every step elementwise) and gives what the earlier attempts
could not:

| | 1 thread | 4 threads |
| --- | --- | --- |
| snake | 0.358 s | **0.099 s** (3.6x) |
| codec | 0.939 s | 0.584 s |
| RTF (int8) | 0.285 | **0.254** |

Same WAV md5. Lesson to carry: before optimizing a function, confirm the process
actually executes it -- an env toggle that changes no timing is evidence the
branch is dead, not evidence the optimization is worthless.

### Codec convolutions: BNNS confirmed over im2col+SGEMM (2026-07-29)

With Snake down to 0.095 s the codec is ~0.45 s of causal convolution. The
existing `MYNAH_CODEC_SGEMM=1` alternative was re-measured on the 6.0 s
utterance and is much worse: codec 1.238 s against 0.593 s, RTF 0.363 against
0.256. BNNS stays the default. Beating it needs a purpose-built threaded causal
conv, not a different library.

### Rejected 2026-07-29: vectorizing Snake

Snake is ~0.31 s, 42% of codec time, and looked like an easy target on the
belief that it ran scalar `sinf`. It does not: `snake_channel` is only the
fallback, and the Accelerate path already uses `vvsinf`. Its real inefficiency
is shape -- a per-call `malloc` and four full passes over a large temporary.
A fused tiled version keeping the temporary in L1 measured only 1.09-1.45x,
about 1% of total runtime, so it was not worth the churn. Recorded so the next
session does not re-derive it. Forcing `MYNAH_SNAKE_SCALAR=1` costs nothing
either (0.313 s both ways), which is itself the tell that Snake is bound by
memory traffic rather than by the transcendental.

On macOS/Accelerate, BNNS is now the default causal codec-convolution path;
`MYNAH_CODEC_SGEMM=1` retains the verified SGEMM/im2col fallback for A/B and
runtime diagnosis. Linux/OpenBLAS and scalar builds are unchanged. The current
BNNS filter API is deprecated by Apple in favor of BNNSGraph on recent SDKs;
keep the narrow diagnostic suppression localized and migrate only when a
BNNSGraph implementation can preserve this numerical/performance result.
Execute this queue in order; reject changes within 3% or outside tolerance.

1. **P0 measurement — done:** `make bench` reuses one loaded model, performs
   explicit warm-ups, reports median synthesis-only RTF; `bench-matrix` runs
   f32/INT8/Q4 with identical prompt, speaker, seed and step cap.
2. **P1 profiler:** add per-stage counters for encoder, decoder prefill, main
   AR, local AR/logits and codec; record calls, bytes and wall time. Capture
   Instruments/Linux `perf` top symbols. Add TTFA, frames/s, peak RSS, CPU
   model, ISA, revision and JSON/TSV output.
3. **P2 correctness gate:** save f32 logits/codes/EOS and WAV metrics; measure
   INT8/Q4 argmax agreement, EOS delta, duration, RMS/peak and mel correlation.
   Keep codec f32. No quantized default before listening and oracle parity.
4. **P3 AR bandwidth:** prepack quantized weights at model-open, align to 64 B,
   reuse activation scratch, fuse QKV and bias, batch the 16 local projections,
   and eliminate tensor-name lookup/allocations in each step. A/B four-row
   SDOT/Q4 unpack, row partitioning and phase-specific thread counts; copy the
   proven `../qwen-tts` patterns, not its dimensions.
5. **P4 codec:** profile each causal conv/transpose stage; cache tensor views,
   reuse all buffers, tile channels, fuse bias/activation where exact, and
   parallelize only stages whose dispatch cost is below 5% of work.
6. **P5 portability/gates:** NEON/SDOT, AVX2 and scalar self-tests; 1/2/4/8
   thread matrix; UBSan plus WAV/oracle regression. Each accepted commit records
   before/after median, variance, RSS and output checksum.

After each phase recompute the remaining budget to 0.1. If measured memory
traffic makes it unattainable on M1, publish the hardware floor and require a
new graph/precision decision rather than relabeling throughput as latency.

### After all CPU + GPU speedups are done (deferred TODO)

Add a single `make` target (e.g. `make gen-matrix`) that synthesizes a short,
light sample — even just "hello world" — for **every one of the 12 supported
languages**, writing one WAV per language into a listenable folder, so users
can quickly audition voice/quality across languages. Keep it lightweight
(short text, fixed speaker/seed) and driven by each language's exported
tokenizer/G2P assets. This depends on wiring the per-language tokenizers
beyond English (byt5 char tokenizers for fr/it/vi/ko are the easiest first
step; phoneme G2P for de/es/hi/ja/pt/zh comes after).

## 19. Checkpoint — 2026-07-24 (session 2): CPU speedups + oracle parity

### CPU speedups accepted this session (M1, MYNAH_THREADS=1, INT8, greedy)

| change | old RTF | new RTF | delta | correctness |
|---|---:|---:|---:|---|
| NEON axpy attn·V (decoder + local) | 0.344 | 0.327 | -4.9% | WAV byte-identical |
| vForce GELU local transformer FFN | 0.331 | 0.320 | -3.3% | WAV byte-identical |
| decoder scratch reuse + pre-resolve | — | — | ~0% | correctness fix (zero alloc/lookup in AR loop) |

Current best: **RTF 0.320** (INT8, M1, 1 thread). Target: ≤ 0.2.

### Rejected this session

- NEON Q·K dot product: flips greedy argmax (24576 vs 29696 samples).
- Q4 quantization: +5–9% slower per stack (compute-bound, unpacking overhead).
- Kernel fusion: max ~1.8% gain (activations are 13% of decoder step).
- Batched matmat: local transformer streams are sequential, not batcheable.
- Metal per-op: RTF 40.7 (13.6 ms dispatch × 3300 calls). Even batched: ~0.48.

### Oracle parity — OPEN ISSUE

NeMo 3.1.0 greedy (temp=0, topk=1, no CFG, no prior) vs native f32,
same token IDs [55,79,90,59,62,87,93,27,39,36,34], speaker 4:

- Oracle: 24576 samples, EOS at step 12.
- Native: 40960 samples, no EOS at step 20.
- **Codes diverge from step 0** (15/16 codes differ).
- By step 7 native generates degenerate repeated-pair codes.
- Some step-0 codes are close (317→316, 1779→1787) → systematic error,
  not a weight conversion failure.

NeMo default inference uses temperature=0.6, topk=80, cfg_scale=2.5
(not greedy). The oracle tool must override these for deterministic
comparison.

**Next step:** dump decoder hidden state (out_last) at step 0 from both
paths to isolate whether the divergence originates in the decoder
(encoder/prefill/AR step) or the local transformer.

`MYNAH_DUMP_CODES=<path>` env var added for per-step codebook JSON dump.

## 20. Oracle parity — resolved (no bug)

Stage-by-stage comparison (NeMo 3.1.0 greedy vs native C, same token
IDs, speaker 4): encoder, baked context, decoder weights, cross-attn
K/V, and GELU all match (max diff < 1e-5).  A manual Python/numpy
recomputation of decoder layer 0 matches the native C with max diff
6e-4, confirming the native is mathematically correct.

The 0.616 max diff at decoder layer 0 between native and oracle is
caused by **Accelerate BLAS vs PyTorch BLAS accumulation order**
(corr(|value|,|diff|) = 0.987).  This 0.12% relative error is
amplified by the local transformer's argmax into different codes
(15/16 differ at step 0) and a different EOS point.

This is not a bug.  It is inherent floating-point non-associativity
across BLAS implementations, consistent with the project contract
("Do not require byte-identical audio across different floating-point
orderings or CPU/GPU backends").

Full analysis: `docs/oracle-parity.md`.
Debug env vars: `MYNAH_DUMP_ENCODER`, `MYNAH_DUMP_PREFILL`,
`MYNAH_DUMP_HIDDEN`, `MYNAH_DUMP_LAYERS`, `MYNAH_DUMP_CODES`.

## 21. Checkpoint — 2026-07-24 (session 2, fine giornata)

### Risultati della sessione

**Performance CPU (M1, 1 thread, INT8, greedy):**
- RTF inizio sessione: ~0.37 → fine: **0.320** (-13.5%)
- NEON axpy attention: -4.9% (WAV byte-identical)
- vForce GELU local transformer: -3.3% (WAV byte-identical)
- Decoder scratch reuse + pre-resolve: zero alloc/lookup nel loop AR

**Oracle parity — risolta (nessun bug):**
- Encoder, baked context, pesi decoder, cross-attn K/V, GELU: tutti MATCH (< 1e-5)
- Calcolo manuale Python/numpy matcha il native C (max diff 6e-4)
- Divergenza vs NeMo = Accumulation order BLAS (Accelerate vs PyTorch), corr 0.987
- Non è un bug: non-associatività FP intrinseca. Docs: `docs/oracle-parity.md`

**Tokenizer nativo C — 12 lingue, zero Python:**
- ByT5 (fr/it/vi/ko): byte+3, EOS=1 — byte-identical a NeMo
- IPA G2P (en/de/es/pt/hi): dizionario + longest-match phoneme parsing
- Arabo (ar): char mapping — byte-identical a NeMo
- Giapponese (ja): hiragana→katakana + vocab lookup (base)
- Cinese (zh): char-level vocab lookup (base)
- CLI: `--text "hello" --lang en`
- `make gen-matrix MODEL_DIR=pack` → 12 WAV, 100% C

**Metal:** self-test PASS, pooled IO + tiled kernel. RTF 40.7 per-op
(non viable per matvec 768-wide; serve full-step GPU o modelli più grandi).

### TODO — prossimo passo

1. **G2P completo per ja/zh:** il tokenizer giapponese serve open_jtalk
   (dizionario ~23MB, analisi morfologica + pitch accent). Il cinese serve
   una tabella character→pinyin (~20K entry) + regole pinyin→fonemi.
   Entrambi fattibili in C ma richiedono l'export dei dati da NeMo.

2. **Allineamento G2P en/de/pt con NeMo:** il C usa sempre il dizionario,
   NeMo ha `phoneme_probability=0.8` stocastico. Per parity esatta servirebbe
   replicare il seeding RNG di NeMo, oppure accettare la differenza (il C
   è deterministico e usa il path principale all'80%).

3. **RTF ≤ 0.2 (release gate):** servono ~0.12s di taglio (37%).
   I guadagni facili sono esauriti su M1. Opzioni:
   - Metal full-step GPU (subgraph residente, non per-op)
   - Modello più piccolo (se disponibile)
   - Q4 selettivo sul local transformer (cambia EOS, va validato a orecchio)

4. **Streaming/long-form (M5):** offline/streaming equivalence, codec
   chunk boundary click-free, flush state machine.

5. **Server/API (M7):** library boundary, concorrenza, HTTP layer.

6. **dots.tts (v2):** GPU-first, 2B parametri, AudioVAE + flow matching.
   Il backend Metal/CUDA ha senso qui.

### Stato dei file

```
src/tokenizer.c/.h    — tokenizer nativo 12 lingue (NUOVO)
src/graph.c           — decoder/encoder/codec + dump debug env vars
src/qmat.c            — INT8/Q4 4-row SDOT matvec
gpu/metal/            — backend Metal (pooled IO, tiled kernel)
docs/oracle-parity.md — analisi parity completa (NUOVO)
tools/gen_matrix.py   — gen-matrix via NeMo (legacy, ora superata da make gen-matrix)
cli/main.c            — --text/--lang/--warmup/--runs
Makefile              — gen-matrix nativo, bench, bench-matrix
```

## TODO — CUDA optimization (sessione DGX2 GB10)

### Risultati
- RTF 14.2 → 0.257 (output corretto, 98.2% più veloce)
- FP16 weights, GPU conv1d, mapped-buffer zero-copy, k_bias_add
- Infrastruttura resident step completa (matmul_d2d, dev_alloc, GPU ops)

### Blocker: tanhf bit-exact GPU (priorità 1)
Il GELU nel local transformer costa 43ms/synthesis (22% del synth).
CPU tanhf (ARM libm) non è riproducibile su CUDA — ogni tentativo
(float, non-FMA, double) cambia EOS detection nel loop autoregressivo.

**Fix richiesto:** reverse-engineer ARM libm tanhf polynomial e
reimplementare in CUDA kernel. Oppure: tabella lookup 4096 entry
con interpolazione lineare (precisione < 0.5 ULP per |x| < 8).

Stima: RTF 0.257 → ~0.20 con GPU GELU bit-exact.

### Fatto
- [x] cuBLAS backend + sgemm
- [x] GPU snake (codec)
- [x] FP16 weight cache
- [x] Mapped-buffer zero-copy output
- [x] k_bias_add kernel (replaces cublasSger)
- [x] GPU conv1d (im2col+sgemm fused, codec -53%)
- [x] Pre-resolved weights (decoder + local transformer)
- [x] Optimized weight extraction (stride-1 reads)
- [x] k_layer_norm bounds check fix (CRITICAL OOB bug)
- [x] Resident step infrastructure (matmul_d2d, dev_alloc, h2d/d2h)

## 22. CPU follow-up queue — 2026-07-26

The next CPU work is ordered by correctness risk and expected measurable value.

### New ideas added

1. Pre-resolve every local-transformer output projection, bias and audio
   embedding tensor at model open; remove per-stream `snprintf()` and tensor
   lookup from the autoregressive loop.
2. Add a qmat activation cache so one quantized input row can be reused by
   projections that consume the same hidden state; compare only within the
   same INT8/EOS configuration.
3. Add a 64-byte-aligned interleaved INT8 weight pack for SDOT, with 4-row and
   8-row layouts selected by matrix shape; retain the unpacked scalar fallback.
4. Add exact residual-plus-LayerNorm fused kernels to reduce memory traffic,
   gated by stage parity before enabling them in the decoder.
5. Add phase-specific parallelism: prefill and large codec stages may use the
   worker pool; one-row AR and sequential local-transformer streams stay on
   one thread.
6. Add AVX2/FMA kernels and cross-ISA golden self-tests before x86 claims.

### CPU TODO order

- [ ] P0: refresh the stale CPU RTF/oracle table and run the powered full-suite gate.
- [x] P1: pre-resolve local output/embedding tensor views.
- [ ] P2: persistent request state and zero-allocation AR path.
- [ ] P3: INT8 activation reuse and aligned interleaved weight packing.
- [ ] P4: residual+LayerNorm fusion with token/code parity.
- [ ] P5: prefill/codec phase-specific threading and AVX2 validation.

### Full-suite TODO when power is available

Run only with the MacBook connected to power: `make test`, scalar/NEON
benchmark A/B with two warm-ups and seven measured runs, `make bench-matrix`,
12-language/speaker matrix, oracle stage dumps, streaming/long-form tests,
`make ubsan`, `make leaks`, RSS/TTFA report, and real-device `make metal`.
Do not use the short battery measurements as accepted performance gates.

### Battery session note

### Architecture TODO — streaming/server split

- [x] Callback streaming emits causal PCM prefixes and preserves offline
  sample parity.
- [ ] Move request state, AR stepping and codec state out of `graph.c` into
  cohesive `inference.c`/`stream.c` modules; keep `graph.c` for tensor graph
  primitives and stage kernels.
- [ ] Replace streaming prefix re-decode with persistent causal-convolution
  and upsampler state, including click-free chunk-boundary tests and zero
  allocations in the AR loop.
- [ ] Define the server-facing lifecycle on top of the same stream state,
  without a second offline/streaming implementation or global request state.

2026-07-26, MacBook at 26%: implemented P1 and ran only `make test`, CPU
self-test, and one cold synthesis (`BENCH_WARMUP=0 BENCH_RUNS=1`). The WAV
was 15,360 samples and byte-identical to the baseline, but the cold RTF 5.09
is intentionally invalid as a performance result. Repeat the full serial A/B
with power connected before accepting or rejecting this optimization.

## 23. Checkpoint — 2026-07-28: Metal resident graph port

The CPU audio-quality issue is resolved; do not describe the CPU output as
gibberish. This session moved the M1 Metal path substantially closer to a
device-resident graph:

- decoder output stays in a Metal buffer when feeding the local transformer;
- local-transformer layers, KV cache, FFN, output projection and constrained
  greedy argmax run on Metal; only the selected token crosses back to C;
- NanoCodec has a resident path for pre/post causal convolution, Snake,
  transposed convolution and residual blocks, with reused device activations;
- device copy/scale/argmax and transposed-convolution backend seams were added;
- standalone Metal parameter buffers now reset correctly outside command batches.

The resident path still downloads logits for temperature/top-k sampling. The
final waveform is clipped and sanitized by a device postprocess kernel, and
the causal-convolution `im2col` workspace is reused across dispatches. Real-M1
validation now passes for the parity-safe hybrid graph and resident codec
(RTF 0.782, exact token/EOS duration, PCM correlation 1.000000000). The
full-device AR graph remains unaccepted because it changes the greedy
trajectory and EOS.

## 24. Server roadmap — ported from the qwen-tts audit (2026-07-29)

`../qwen-tts` has a far more exercised server (54 KB against our ~15 KB, nine
server test targets against our one). This section records what that audit found
worth taking, what is blocked, and what does not apply — so the next session
does not re-derive the comparison.

Execute in the order below; each item is independently shippable.

### P0 — robustness gaps in our server — DONE 2026-07-29

1. **Worker pool with a bounded connection queue.** Our server calls
   `pthread_create` per connection with no ceiling: N connections means N
   threads. qwen uses `accept → conn_queue → fixed worker pool`. This is the
   most serious defect in the current server and it is trivially reachable by a
   client that just opens sockets.
2. **`SO_RCVTIMEO` on accepted sockets.** A client that connects and never
   sends leaves our reader blocked in `recv` forever. Combined with (1) that is
   a denial of service with no exploit needed, just a loop.

### P1 — the tests that guard *our* specific risk — DONE 2026-07-29

3. **Reproducibility test.** qwen's `test-serve-repro` asserts three identical
   requests return bit-identical audio. They added it after a real bug: a stale
   `ctx->dec_x` surviving between requests. Our `mynah_tts_model` carries
   exactly that shape of hazard — `qcache`, `codec_cache` and
   `local_projection_cache` are mutable and populated during synthesis — so this
   is the highest-value test to port. `tests/test_server.sh` currently checks
   stream/batch parity but never checks that the *same* request twice agrees.
4. **Concurrency test.** Two concurrent clients must both complete with correct
   audio. Our synthesis is serialized, so this asserts the mutex actually holds
   rather than assuming it.

### P2 — cheap usability — DONE 2026-07-29

5. **Playback recipes in `docs/server.md`** — `ffplay`/`sox`/`aplay` one-liners
   so the streaming endpoint can be heard, not just written to disk.
6. **Expose the remaining sampling params.** `mynah_tts_request` already carries
   `topk` and `max_steps`; the server only reads `temperature` and `seed`.
7. **A native `/v1/tts` route** beside the OpenAI one, with honest field names
   (`text`/`speaker`/`language` instead of `input`/`voice`). qwen keeps both.

**P0/P1/P2 outcome.** The pool (`-w`, default 4) drains a 256-deep queue and
answers 503 with `Retry-After` when full; accepted sockets carry a 30 s
send/receive timeout. `make server-test` now also asserts three identical
requests return byte-identical audio, that two concurrent clients both complete
without perturbing each other, and that `/v1/tts` matches `/v1/audio/speech`.

Worth recording: **the reproducibility test passes as written.** The stale-state
bug qwen-tts hit does not reproduce here — our per-request parameters are already
isolated and the model's mutable caches are populated idempotently. The test is
kept as a regression guard, not as a fix, and it is now the thing that will catch
this if the per-request-context refactor below ever gets it wrong.

### P3 — audit corrected: the blocker is narrower than assumed (2026-07-29)

The claim that concurrency needs "the scratch moved into a per-request context"
was based on the *presence* of three caches on `mynah_tts_model`, not on what
they hold. Inspecting them changes the picture:

| state | actually |
| --- | --- |
| `local_projection_cache` | built at model open, read as `const` — already safe |
| `codec_cache` | **already carries a `pthread_mutex_t`**; holds BNNS filters and tap entries, i.e. derived read-only data |
| `decoder_cache`, `local_workspace` | allocated and freed per synthesis call — already per-request |
| `qmat` cache | **the only unguarded shared mutable state** |

No mutable statics exist in `graph.c`, `backend.c` or `kernels.c`.

**Fixed:** the qmat cache is now mutex-guarded, and its entries moved to
indirect storage. A lock around get-or-create alone would not have been enough:
the entry array was grown with `realloc`, which *moves* it, while another thread
holds an entry pointer for the whole duration of its matvec — that pointer would
dangle. Entries are now individually allocated, published only once fully built,
and never mutated afterwards, so readers need no lock at all.

**The BNNS question, settled — and the diagnosis was wrong twice.**

First correction: the SDK header does *not* say `BNNSFilterApply` is unsafe on
multiple threads. The one thread warning in `bnns.h` (line ~2240) is attached to
`BNNSFilterGetDataPointer`, about mutating filter parameters while others use
the filter. `BNNSFilterApply`'s thread-safety is simply **undocumented**. The
newer `bnns_graph.h` does state "the same context must only be used by a single
thread at a time", which is suggestive but is about a different API. Do not
quote the old warning as if it covered apply.

Second correction, and the one that mattered: `BNNSFilterApply` was **already
inside** the cache mutex (lock before create, unlock after apply), so concurrent
apply on one filter could never happen. The real defect was elsewhere — the
**lookup loop scanned `cache->entries` without holding the lock**, while the
insert path grew that array with `realloc` under it. Unlocked scan against a
concurrent move: the same class of bug as the qmat one, found only by reading
the lock boundaries rather than trusting the mutex's presence.

**Fixed:** the lookup now holds the lock, and entries are keyed by owning thread
(`pthread_equal`). Each thread therefore has its own filter for a given shape,
which sidesteps the undocumented question entirely *and* lets the apply run
outside the critical section. Previously the whole convolution sat inside the
lock, so the codec was serialized process-wide even where nothing required it.

Cost: filters are allocated per (shape, thread), bounded by the worker pool
(4 by default). Output is unchanged — the f32 WAV md5 still matches the session
baseline.

Secondary: `mynah_parallel_for` runs a region inline when the pool is already
busy, so two concurrent synthesis calls would each lose their inner
parallelism. Correct, but it caps the benefit until the pool is per-context.

8. **Concurrent synthesis with cloned contexts.** qwen clones its ctx per worker
   sharing read-only weights. For us this needs `qcache`/`codec_cache`/
   `local_projection_cache` moved off `mynah_tts_model` into a per-request
   context. Until then extra synthesis threads would corrupt output, not speed
   it up, which is why the server serializes behind one mutex today.
9. **Continuous request batching (vLLM-style).** One scheduler owns the model,
   admits queued requests into free slots, steps them together frame by frame
   with per-slot KV and RNG. Weight-stationary: the weights are read once for
   all in-flight requests.

   This is the *right* lever for us specifically. Decode is DRAM-bandwidth bound
   (see section 21), so N concurrent requests would cost roughly the bandwidth
   of one. It is the largest available throughput win in the project — and it is
   gated entirely on item 8.

### Deliberately not ported

`emotion`, `instruct`, inline `[joy]`/`[sad]` markup and voice cloning all
depend on capabilities **Magpie v2607 does not have**: no steering vector, no
instruct conditioning, no zero-shot cloning. Porting them would mean inventing
model behavior. `volume` is a trivial linear gain if ever wanted; `rate` needs a
pitch-preserving resampler, which is real DSP rather than a port.

### P3-9 continuous batching — premise validated by measurement (2026-07-29)

Before opening a large refactor of graph.c, the premise was tested directly:
does batching N decode requests cost one bandwidth pass, or N?

Measured on M1, qkv shape 2304x768, 198 MB working set (far beyond any cache),
best of 3. "Serialized" = each request walks the entire weight set in turn, which
is what the server does today. "Batched" = one walk with B rows in flight.

| B | serialized | batched | speedup | per request |
|---|---|---|---|---|
| 1 | 6 ms | 6 ms | 0.93x | 6.2 ms |
| 2 | 11 ms | 10 ms | 1.13x | 5.0 ms |
| 4 | 22 ms | 9 ms | 2.32x | 2.3 ms |
| 8 | 43 ms | 10 ms | **4.46x** | **1.2 ms** |

Batched time is flat while serialized scales linearly. Per-request cost falls
5x at B=8. The premise holds: **N concurrent requests cost approximately the
bandwidth of one**, because each weight is read once and reused across all rows.

A first attempt at this measurement was wrong and is worth recording. It looped
B single-row calls over the *same* matrix, which stays in cache after the first
pass, so it measured cache reuse rather than the decision being made — and
produced incoherent speedups (2.04x, 1.05x, 1.60x for one shape). The fix was to
make the serialized arm walk the *whole* set once per request, which is what the
decoder actually does: a request traverses every layer before the next begins,
so nothing survives in cache between requests.

**Implementation shape** (multi-session; nothing partial is safely shippable,
because a half-converted AR loop is incorrect rather than slow):

1. Lift the AR step from single-row to B-row: the decoder and local-transformer
   projections become matmat with B rows. `mynah_backend_matmul` already takes
   `rows`, so the kernels need no change — the state does.
2. Per-slot state: KV caches, RNG, sampling params, EOS flag. `decoder_cache`
   and `local_cache` are already per-call, so they become per-slot arrays.
3. A scheduler owning the model: admit queued requests into free slots, step all
   active slots together, finalize on per-slot EOS, refill the freed slot
   immediately (continuous, not static batching).
4. Ragged handling: slots are at different positions, so attention must mask per
   slot rather than assume one shared length.
5. The server drops its synthesis mutex and hands requests to the scheduler.

Gate: with B=1 the output must stay byte-identical to today's, and
`make server-test`'s reproducibility check must keep passing under batching.

### P3-9 payoff corrected by phase measurement (2026-07-29, later)

The 4.46x above is real but it is the speedup **of the matvec**, not of the
product.  Before opening the refactor the phase split was measured, and it does
not support the implied end-to-end number.  Long text, M1, int8, instrumented:

| phase | share |
|---|---|
| prep (text encode + context prefill) | ~20% |
| AR decoder step | ~20% |
| AR local transformer | ~31% |
| codec | ~29% |

The AR decoder step -- exactly what steps 1-4 above convert -- is about a fifth
of wall time.  Amdahl on the measured 4.46x gives `1/(0.8 + 0.2/4.46)` =
**1.22x** throughput at B=8, not 4.46x.  Batching the local transformer too
(both are `mynah_qmat_linear_resolved` call sites with per-slot state, and per
the note below their attention does *not* need batching) covers ~51% and gives
**1.65x**.  Reaching anything near 4x additionally requires batching the codec.

One simplification found while measuring, worth keeping if this is ever built:
**only the weight-reading ops need to batch.** Attention reads each slot's own
KV cache, which is private memory shared with nobody, so batching it buys
nothing -- it stays a per-slot loop with its own length, exactly as today.  That
removes the ragged-masking problem in step 4 entirely.

Also note the numerics.  A B-row sgemm is not bit-identical to B single-row
calls, so timing-dependent batch composition would make the same request return
different audio depending on who it batched with -- a visible nondeterminism
regression against `server-test`'s reproducibility check.  The quantized
kernels can avoid this (we control the per-(row, b) accumulation order, so a
weight-stationary loop over row blocks is bit-exact), but the f32 Accelerate
path cannot and would have to stay unbatched.

Two cheaper levers were tested against this and both are closed:

- **int4** instead of int8: RTF 0.229 vs 0.241, ranges overlapping.  ~5%, and
  it perturbs the trajectory.  int8 stays the default.
- **f16**: RTF 0.330, clearly worse -- 2 weight bytes against int8's 1 on a
  bandwidth-bound decode.

Verdict: P3-9 is a throughput-only win of roughly 1.65x at B=8 for a large,
multi-session refactor of the decoder *and* local transformer.  It is no longer
the highest-value item; recording this so the 4.46x is not quoted again.

### P3-9 DONE (2026-07-29) — commit 8c7abc2

Implemented and measured. 1.63x aggregate throughput at 8 in flight, every
batched request byte-identical to the same request run alone, single-request
output unchanged.

What differed from the plan above:

1. **Attention needed no batching and no ragged masking.** Step 4 of the
   original plan was wrong-headed: attention reads each slot's private KV
   cache, shared with nobody, so batching it amortizes nothing. Keeping it per
   slot means each simply loops over its own length -- the ragged problem
   dissolves instead of being solved.
2. **The local transformer mattered more than the decoder.** It runs once per
   stacked stream, so one decode step walks its weights sixteen times: 252 MB
   against the decoder's 99 MB at int8. The plan only mentioned the decoder.
3. **Bit-exactness turned out to be achievable, and load-bearing.** The plan
   gated on "B=1 byte-identical"; that is too weak. Without exactness at every
   B, a request's audio depends on who it batched with. The quantized kernels
   give it because we control the per-(row, b) accumulation order; f32 sgemm
   does not, so that path stays unbatched.
4. **No separate scheduler owning the model.** synthesize_stream became one job
   through the slot driver, so there is a single implementation rather than a
   batched one alongside a single one.

Ceiling reached: 1.63x, capped by Amdahl at the ~51% of wall time that the
decoder plus local transformer occupy. Next lever for throughput is the codec
(~29%); next lever for latency is unrelated to batching.

### P3-10 codec: measured and closed as not worth it (2026-07-29)

The codec is ~29% of wall time but is compute-bound, not bandwidth-bound: its
405 MB of weights are amortized over the whole sequence, so cross-request
batching is worth ~1%. Threading its convolutions on our pool was implemented
and measured slower (0.430 s -> 0.536 s) because BNNS already parallelizes them;
`MYNAH_THREADS` sizes our pool, not BNNS's. Both reverted/not pursued, written up
in docs/performance.md so it is not retried.

Remaining levers are kernel-level, not scheduling-level.

### P3-11 codec kernels — the open idea (not started)

The only codec lever left is the kernels themselves, not scheduling. Scope this
before writing anything, because BNNS is a real opponent and beating it is not a
given.

Where the time is (M1, int8, 50 steps, 4 threads, codec = 0.430 s):
stages [0.062 0.091 0.118 0.086 0.071], of which snake 0.068 s and transpose
0.007 s already scale; the convolutions are ~0.353 s and are BNNS-internal.
Stage 2 is the largest single stage.

Questions to answer first, in this order:

1. What is BNNS actually achieving? Compute the FLOPs of each stage's
   convolutions and divide by its measured time. If it is already near the
   machine's FP32 peak, stop -- there is nothing to win and that answer is worth
   having written down.
2. If it is far from peak, why? The suspects are the layout (BNNS wants
   ImageCHW and we hand it channel-major frames), the dilated convolutions in
   the residual branches (kernels 3/7/11 at dilations 1/3/5, which vectorize
   badly), and the fact that these are 1-D convolutions expressed as 2-D with a
   height of 1.
3. Only then consider an im2col + sgemm path. `MYNAH_CODEC_SGEMM` already
   selects one, and `MYNAH_CONV_TAP_GEMMS` a tap-accumulating variant -- measure
   those against BNNS per stage before writing a third. The encoder conv-FFN
   work (commit 3206f14) showed the weight layout is already contiguous in
   (in * kernel + k), so an im2col over the activation needs no weight copy.

Gate, same as everywhere else: byte-identical audio, or a documented tolerance
against the oracle if the reassociation is deliberate.

Do not retry cross-request batching or pool-threading the convolutions -- both
measured, both closed, see docs/performance.md.
