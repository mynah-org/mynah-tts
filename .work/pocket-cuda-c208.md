# Pocket 24L CUDA on the L4: towards C208

Branch `pocket-cuda-c208` from main 5ae4fd2 (PR #6 merged). Goal: qualify C208 on one L4 (sm_89, 24 GB) with the
24L English model, the level a reference PyTorch implementation of the same model reaches on the same GPU, and go
past it if possible. Method: 60-s closed-loop knees at C160/C192/C208, interleaved A/B arms on one box, then 2-min and
10-min soaks with the usual gates (stream RTF p95 < 0.88, no stall@500, TTFA p95 <= 500 ms).

## Where the step goes at C160 (nsys, 2026-10-02)

nsys 2024.6 with `--cuda-graph-trace=node`, 10 s window under a C160 closed loop, main 5ae4fd2 defaults
(KV bf16, SEANet bf16, slot-pool prefill, width buckets). About 155 steps in the window, so the step is ~64 ms on an
80 ms frame (stream RTF p95 0.816).

| stage | ms per step | what |
|---|---|---|
| decode attention `k_self_attention_bf16_batch_fast` | 15.7 | bandwidth bound: every row reads its whole K/V (~4.7 GB per step), voice prefix included |
| SEANet decoder (gang) | ~17 | GEMMs ~5 ms; elementwise and layout kernels ~12.6 ms (im2col 3.1, ELU 2.3, convtr overlap 1.8, window 1.8, residual 1.6, copy prefix 0.9, bias 0.7) |
| text prefill `k_tile_gemm_part` (SIMT fixed-order) | 8.7 | no tensor cores |
| backbone GEMMs (cuBLAS TF32) | ~7 | |

NVTX: `step.backbone` 23.7 ms, `codec.total` 22.7 ms, `step.embed` (prefill) 9.9 ms mean / 27 ms max.

Note: without `--cuda-graph-trace=node` nsys reports graph launches only, so the backbone and decoder kernels are
missing from the kernel summary. And `--kill none` leaves the server running after the capture: stop it explicitly
before the next arm, or the next health check answers from the old process.

## Changes on this branch

### `MYNAH_CUDA_SHARED_VOICE` (default 0 while measured)
The batched decode attention reads positions `[0, voice_positions)` of each row from the model-owned device voice
cache (`state->cuda_voice_kv[speaker]`, `[layer][K|V][T][attn]`) instead of the row's own copy. The prefill tile still
copies the prefix into the row, so the values are the same bf16 numbers and every other reader (tile attention,
KV growth, host mirror) is unchanged; any row or configuration the fast kernel cannot take reads the row as before.
Rows of one voice then read one copy that stays in L2 (0.5 MB per layer and voice) instead of streaming it from DRAM
once per row.
- Plumbing: `mynah_backend_self_attention_bf16_prefix_batch_dev`, per-layer prefix pointer tables and a per-row
  prefix length next to the existing K/V tables (same graph-replayed host tables).
- Bit identity (L4, 24 seeded requests in flight, 4 voices, SEANet fp32): off, off again and on give the same md5 for
  all 24; `--gpu-self-test` and `--pocket-self-check` PASS with the flag on.
- Next if it pays: drop the per-row copy and the prefix positions from the row allocation (-12 MB per row, ~2.4 GB at
  200 rows), which needs the prefill tile attention and the KV growth copy to know about the prefix too.

### `MYNAH_CUDA_DECODER_FUSE` (default 0 while measured)
In the cross-request decoder, ELU and the causal window are folded into the kernels that read them:
`k_decoder_causal_columns_fused` builds the im2col columns straight from the carried state and the input (ELU on the
fly), `k_decoder_copy_tail_fused` moves the carried state from the input (only when tail <= length, which holds for
every SEANet op), and the transposed-convolution gather applies the pre-ELU. Same float operations in the same order,
so the audio should be bit-identical; the ELU output and the window buffer are no longer written. Flag off: the same
kernel sequence as before.

### Decoder fusion, part 2 (same flag, `MYNAH_CUDA_DECODER_FUSE`)
Part 1 removed the ELU and causal-window passes but the knee did not move, so the rest of the glue goes too. All of
it is in the cross-request (gang) decoder only (`decoder_step_batch_impl`); the solo decoder is untouched, and with
the flag off every launch, GEMM argument and buffer is the old one (the only flag-dependent bookkeeping is the
per-graph upload bound of a tail-free conv, 4 -> 6 table slots, and only with the flag on).

Kernels per decoder op (gang path, SEANet bf16, GEMMs counted apart; ELU/window already gone with part 1):

| op | part 1 | part 2 |
|---|---|---|
| first conv (k7) | columns, copy_tail, bias + GEMM(beta 1) | columns, copy_tail + GEMM(beta 0); bias -> next gather |
| ELU + transposed conv (GEMM form) | gather, GEMM, overlap (writes `full`), fold, copy_prefix | gather (+producer bias or residual), GEMM, overlap_out |
| resblock conv1 (k3) | columns, copy_tail, bias + GEMM(beta 1) | columns, copy_tail + GEMM(beta 0); bias -> conv2 columns |
| resblock conv2 (1x1) | columns, bias + GEMM(beta 1) | columns (+conv1 bias) + GEMM(beta 0); bias -> residual |
| residual add | residual (reads x and y, writes x) | none: x + (y + b) is resolved by the next reader |
| last conv (k3, ELU) | columns, copy_tail, bias + GEMM(beta 1) | columns, copy_tail (+residual), bias + GEMM(beta 1) |

For a topology of n upsampling stages with one residual block each that is 6 + 10n elementwise/layout launches per
step before and 5 + 5n after (n = 3: 36 -> 20), and the passes that went were whole-tensor ones: the residual
(2 reads + 1 write), the bias fill plus the GEMM's read of C, the `full` write + reread and the prefix copy. If a
residual block is followed by another one (n_residual_layers > 1), or the reader cannot take the fused path (one-GEMM
mode, a conv with tail > length, the SIMT transposed conv), the sum is written as before
(`k_decoder_residual_bias_batch` when the 1x1 bias was deferred) and nothing is deferred into that reader.

Why each one is exact (same fp32 operations, same operands, same order):
- Deferred bias. Before: `k_decoder_bias_batch` writes C = bias (0.0f when the op has none), GEMM alpha = beta = 1.
  With alpha = beta = 1 every epilogue form (alpha*acc + beta*C, fma(alpha, acc, beta*C), fma(beta, C, alpha*acc))
  is the single rounding round(acc + C), because multiplying by 1 is exact. After: GEMM beta = 0 stores acc, and the
  reader computes `acc + (bias ? bias[c] : 0.0f)` in fp32 before anything else (the +0.0f keeps -0 -> +0 as before).
  The one assumption is on cuBLAS's side: that it runs the same algorithm, accumulating acc in the same order, for
  beta = 0 and beta = 1. A serial split-K kernel that folds C into the first partial, or a heuristic that picks a
  different kernel for beta = 0, would break bit identity (not correctness). cuBLAS does not document either way,
  so this part can be switched off alone: `MYNAH_CUDA_DECODER_FUSE_BIAS=0` keeps bias + beta = 1 and leaves the
  rest of the fusion on. Also assumed: the cuBLAS epilogue does not flush subnormals (our kernels build without
  fast math, so they do not), which only matters for |acc + bias| < 1.2e-38.
- Residual. Before: y = round(acc + b) (GEMM epilogue), then x = round(x + y) (`base += add`). After: the reader
  (`decoder_lazy_value`) computes v = acc + b, then x + v, then the ELU, i.e. the same two additions on the same
  operands; fp32 addition is commutative bit for bit, and there is no multiply for the compiler to contract into an
  FMA. Readers: the transposed-conv gather and, for the last stage, the last conv's column and carried-state kernels
  (each tap and the state copy recompute the same value from the same inputs, so every copy is the same bits).
- Transposed conv. Before: overlap wrote full[t] = bias + taps (fixed order), the fold did full[pos] += partial[pos]
  and partial[pos] = full[output_len + pos] - bias, and copy_prefix moved full[0, output_len) to the destination.
  After: `k_decoder_convtr_overlap_out` sums the taps of t with the same helper loop (bias first, taps in the same
  order), adds partial[t] for t < tail, and the thread of t < tail also sums sample output_len + t and stores it
  minus the bias as the new partial. That thread is the only reader and writer of partial[t]; with
  tail <= output_len (tail = stride for SEANet, output_len = length * stride) no fold write lands on a sample the
  fold reads, which is what the old two-kernel order guaranteed. `full` is no longer written (it was scratch only).
- Pointer tables: the fused conv uploads state, columns, input, residual (channel 3, before the weight table
  replaces it), output and weight tables, at most 6 per conv and 4 per transposed conv, under the existing
  per-op bound (9 / 8 with ELU slots); the tail-free conv bound is raised to 6 with the flag on. In-place aliasing:
  every reader of a deferred value (columns, carried state, gather) is launched before the op's first write to its
  destination (bias fill or GEMM / overlap_out), and the destination of the op after a residual block is the buffer
  holding y, written only after the gather / column kernels have read it.

Gate caveat: `--pocket-self-check` in the pedantic configuration is a 1e-4 abs/rel tolerance check (gang vs solo),
not a bit comparison (and in that configuration the gang transposed conv is a GEMM while the solo one is the SIMT
kernel, so they are not bit-equal anyway). Bit identity of part 2 has to be shown as gang output md5, flag off vs on
with the same gang history, once with `MYNAH_CUDA_DECODER_FUSE_BIAS=0` (pure elementwise fusion, should match by
construction) and once with the default (also tests the cuBLAS beta = 0 assumption).

## Measurements
(filled in as the runs land)

### 2026-10-02 knee A/B (60 s closed loop, 4 voices, server --max-batch 208, two interleaved rounds)
Stream RTF p95, round 1 / round 2:

| arm | C160 | C192 | C208 | audio-s/s at C208 |
|---|---|---|---|---|
| main 5ae4fd2 defaults | 0.811 / 0.818 | 0.947 / 0.949 | 1.031 / 1.038 (stalls) | 163 |
| `MYNAH_CUDA_SHARED_VOICE=1` | 0.731 / 0.733 | 0.845 / 0.847 | 0.921 / 0.922 (0 stalls) | 185 |
| `MYNAH_CUDA_TILE_TC=1` | 0.792 / 0.793 | 0.919 / 0.917 | 1.000 / 1.001 | 166 |
| `MYNAH_CUDA_PREFILL_FIXED=0` (cuBLAS prefill) | 0.757 / 0.757 | 0.878 / 0.879 | 0.960 / 0.959 | 174 |

Shared voice: -10/11 % RTF p95, +13 % audio-s/s, bit-identical audio. The cuBLAS prefill shows the prefill lever
(-7 %) but gives up the chunked-equals-whole invariance; the TF32 fixed-order tile gets a third of it. Hence
`MYNAH_CUDA_PREFILL_BF16TC`: the prefill is bound by reading every layer's weights once per call (~1.2 GB in fp32,
~4 ms on the L4), so a BF16 resident copy on BF16 tensor cores in the same fixed order should reach at least the
cuBLAS number and keep the invariance.

### 2026-10-02 correctness
- Pedantic self-checks (`MYNAH_CUDA_TF32=0 MYNAH_CUDA_SEANET_BF16=0`: the backend reports batch-invariant, so
  `--pocket-self-check` compares bits, gang decoder vs solo decoder and a text pushed in pieces vs whole): PASS with
  `DECODER_FUSE`, with `SHARED_VOICE`, with `PREFILL_BF16TC`, and with all three. Each flag's start-up line appears, so
  the new paths really ran. The fused decoder is exact; the BF16 prefill tile is batch-invariant.
- Under concurrent serving load the default configuration is not batch-invariant (TF32 cuBLAS decode GEMMs pick
  their algorithm by width): the same 24 seeded requests at concurrency 24 and 3 differ in all 24 md5s, flags off.
  So md5 comparisons between two server runs only hold when the gang history is the same; at temperature 0 two runs
  of one build are usually identical (99 dB) but one request in 24 can diverge (24 dB) from timing alone.
