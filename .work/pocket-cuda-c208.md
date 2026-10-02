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

### Phase 2: rows without the prefix (`MYNAH_CUDA_SHARED_VOICE=1`, `MYNAH_CUDA_SHARED_VOICE_STRIP` default 1)
With the shared voice on, a row whose prefix every reader can take from the device voice cache no longer stores it:
its backbone KV shrinks by `P = voice_positions` (~126 positions x 24 layers x 2 planes x 1024 x 2 bytes = ~12.4 MB
per row, ~2.5 GB at 200 rows). The C224 OOM (peak 22.5 GB of 23 during KV growth) is the target.
`MYNAH_CUDA_SHARED_VOICE_STRIP=0` keeps phase 1 (prefix still copied into every row) for an A/B in one build.

Design (explicit skip, least invasive):
- `ctx->cuda_backbone_kv_skip = S` (0 or P), fixed at allocation by `pocket_cuda_kv_skip_planned`: shared voice
  on, BF16 KV, prefill tile path predicted (same clauses as `pocket_cuda_prefill_tile_usable`), backend has the
  tile and both prefix-aware attentions, voice non-empty and smaller than the cache. Everything else keeps S = 0,
  the old layout byte for byte (voice cloning without a model voice entry, file-backed voices, f32 KV, CPU,
  `MYNAH_CUDA_PREFILL_TILE=0`).
- Layout with S > 0: `[layer][K|V][capacity - S][attn]`, position p >= S at plane slot p - S.
  `cuda_backbone_capacity` still counts absolute positions, so every "needed <= capacity" check, the growth
  trigger and the step pre-flight are unchanged.
- Decode kernels index `base + position * stride`; they get a plane base biased S positions below the allocation
  (`pocket_cuda_kv_position_base`, integer arithmetic). It is never dereferenced below S: every decode kernel reads
  [0, S) from the shared planes and writes only the current position (>= S). With S = 0 the pointer is exactly the
  old one.
- The prefill tile cannot use a bias (slot = absolute % ring), so `mynah_backend_tile_desc` gained `skip[rows]` and
  `prefix[rows * layers]`: slot = (absolute - skip) % ring with ring = stored slots, positions below the skip read
  from the shared planes. NULL skip = the plain call; the CUDA driver then launches the `PREFIX=false` template
  instances, which are the old kernels, with the old staging layout.

Every reader/writer of the row KV:
| site | change |
|---|---|
| `pocket_cuda_backbone_alloc` | sizes by stored positions; slot reuse lays a parked cache out as stored + S |
| slot pool (`pocket_cuda_slot_acquire`/`_kv_fits`) | byte-based, unchanged: a stripped row simply needs fewer bytes |
| `pocket_cuda_backbone_reserve` (KV growth) | plane sizes and the live count in stored positions ([S, offset)) |
| `pocket_cuda_backbone_upload` | refuses S > 0 (such a row is device-owned, never host-seeded) |
| `pocket_cuda_prefill_tile` | no voice D2D copy for S > 0, only records the shared entry; kv = stored plane starts, rings = stored, skip/prefix tables |
| `k_tile_rope_store`, `k_tile_attention`, `k_tile_attention_grouped` | `<KV, PREFIX>` templates; PREFIX reads [0, S) from the shared planes |
| batched decode (`pocket_cuda_backbone_step_batch`) | biased K/V tables; a stripped row without its shared entry is "not eligible" |
| `k_self_attention_bf16_batch_fast<true>` | unchanged (already reads [0, S) from the prefix and writes position >= S) |
| `k_self_attention_bf16_batch` (legacy fallback) | now `<SHARED>`; with prefix tables the wrapper never falls back to a kernel that reads the row below S (alignment miss or `MYNAH_CUDA_BACKBONE_ATTN=legacy` take `legacy<true>`) |
| single-row step (`pocket_cuda_backbone_step`) | new `mynah_backend_self_attention_bf16_prefix_dev` = `k_self_attention_bf16<true>`, same kernel and order as the plain single-row call; host mirror of the new slot via the biased base |
| `k_gather_kv*_batch` (host mirror) | unchanged: reads only `position >= S` |
| padding rows (`cuda_pad_kv`) | unchanged: own one-position cache, prefix_len 0 |
| `pocket_seed_backbone` non-tile path | a planned S > 0 row that does not take the tile drops its device cache and runs on the CPU (not expected) |

To verify on the GPU (L4):
1. Bit identity, pedantic: `MYNAH_CUDA_TF32=0 MYNAH_CUDA_SEANET_BF16=0 MYNAH_CUDA_SHARED_VOICE=1 --pocket-self-check`
   must PASS bitwise, and print both start-up lines ("decode attention reads voice prefixes ..." and
   "backbone KV rows do not store the N-position voice prefix"). Same with `MYNAH_CUDA_SHARED_VOICE_STRIP=0`.
   `--gpu-self-test` PASS.
2. Same 24 seeded requests (4 voices, SEANet fp32, temperature 0, same concurrency): md5 identical for
   flag off / `SHARED_VOICE=1 STRIP=0` / `SHARED_VOICE=1` (strip). Also with `MYNAH_CUDA_KV_GROW_INITIAL_STEPS=8`
   (forces growth mid-request), `MYNAH_CUDA_BACKBONE_ATTN=legacy` (fallback kernel with prefix),
   `MYNAH_CUDA_TILE_ATTN_GROUPED=0` (non-grouped tile attention) and a single request alone (single-row step).
   Long-form/appended text (prefill in pieces) and multi-segment requests (segment prologue reseeds the row).
3. `compute-sanitizer --tool memcheck` on a short C4 run with the strip on: no out-of-bounds (the biased bases are
   never dereferenced below the skip).
4. VRAM: `nvidia-smi` peak at C208 for off / STRIP=0 / strip (expect ~-2.5 GB with strip), then the C224 knee
   (server `--max-batch 224`) with strip: no OOM during KV growth, stream RTF p95 and stalls.

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

### Decode attention, split kernel (`MYNAH_CUDA_ATTN_SPLIT=1`, default 0 while measured)
`k_self_attention_bf16_batch_split<SHARED, BF16OUT>` (`gpu/cuda/backend_cuda.cu`) is a flash-decoding layout of the
batched BF16-KV decode attention. Same signature, tables and semantics as `k_self_attention_bf16_batch_fast`: it
writes the new K/V at `position` first (then `__syncthreads`), reads positions `< prefix_len` from the shared voice
planes (stride `width`) and never dereferences a (possibly stripped/biased) row below the prefix, and `BF16OUT`
stores the RNE BF16 of the FP32 result into the staged activation. The wrapper picks it wherever the fast kernel
would run (same alignment/stride checks, same fallbacks to the legacy kernel) and `head_width == 64`; otherwise the
old kernels run unchanged. One start-up line: "CUDA decode attention uses the split (flash-decoding) kernel".

Design (one 128-thread block per (head, row), 4 warps, grid unchanged `heads x rows`):
- Lane group g = lane / 8 (4 per warp) owns one position, lane sub = lane % 8 owns dims [8 sub, 8 sub + 8). One
  `uint4` load per lane reads the full 128-byte K (or V) head row of a position, so one warp load instruction
  fetches 4 positions' rows, fully coalesced per 128-byte line.
- Warp w walks positions [16 w + 64 k, 16 w + 64 k + 16): per step each lane issues 4 K and 4 V 16-byte loads
  before any math (8 x 16 B in flight per lane, ~4 KB per warp), then the dot (8 FMA + 3 xor-shuffles inside the
  8-lane group), one running-max update per 4 positions (1 + 4 `expf`) and V accumulation into 8 FP32 registers.
  No block barrier inside the loop; the prefix/row choice is per position, so a prefix boundary inside a step is
  handled, and lanes past `n` load nothing.
- Merge: the 4 groups of a warp by xor-shuffle (8, then 16), then the 4 warps in shared memory in warp order
  (threads 0..63, one output dim each). Warp 0 always holds position 0; an empty warp has weight 0.
- Determinism: the partition and merge order depend only on the row's own length, no atomics, so the result is
  deterministic and independent of the batch composition. It differs from the fast kernel in the last bits
  (different summation order); FP32 math throughout (`expf`, no fast-math intrinsics).

Expected bandwidth. Bytes per step = rows x 24 layers x 16 heads x n x 256 B (K + V, 128 B each per position and
head) = rows x n x 96 KiB. At C160 with mean n ~300: ~4.7 GB, i.e. ~15.7 ms at the L4's ~300 GB/s, which is what
the fast kernel already took before the shared voice: without the shared voice the old kernel is at the DRAM
roofline and the split kernel cannot win much. The case it targets is `MYNAH_CUDA_SHARED_VOICE=1`: the ~126-position
prefix (~2 GB per step at C160) comes from L2, the DRAM part is the suffix only (~160 x 174 x 96 KiB = ~2.7 GB,
~9-10 ms at ~280 GB/s achievable), and there the old kernel (one position per thread, 2-byte V loads, 3 barriers
per 128 positions) is latency/issue bound rather than DRAM bound. Floor at C160 with shared voice: ~10 ms DRAM +
~1 ms L2. Going below needs fewer bytes (8-bit suffix KV, shorter segments), not a better kernel. Occupancy: ~80
registers per thread expected, ~6 blocks (24 warps) per SM, ~100 KB of loads in flight per SM, far above the
~3 KB per SM Little's law needs at 300 GB/s.

Self-test (`--gpu-self-test`, always run, flag-independent): `cuda_attn_split_self_test`, 8 rows x 4 heads x 64,
positions {0, 1, 15, 64, 127, 300, 701, 2049}, strides width and width + 64, prefix lengths {0, 0, 0, 40, 126, 126,
126, 0}, SHARED off and on. Checks: split vs fast FP32 within 1e-3 relative (max(1, |ref|)); BF16OUT within 1e-2;
each row launched alone bit-identical to the same row in the batch; with SHARED the row's own positions below the
prefix are filled with NaN, so a read there fails. With `MYNAH_CUDA_ATTN_SPLIT` set it also prints the max
relative diff. The algorithm was checked on the CPU by a lane-level emulation against a float64 softmax for n in
1..2050 and prefix lengths 0/40/126 (max rel error 4e-7); the CUDA code itself was syntax-checked with clang
(host and sm_89 device) but not compiled with nvcc here.

How to A/B on the L4:
1. Build, `nvcc -Xptxas -v` (or `cuobjdump --dump-resource-usage`) on `k_self_attention_bf16_batch_split*`: no
   spills, registers <= ~96.
2. `MYNAH_CUDA_ATTN_SPLIT=1 ./mynah-tts --gpu-self-test` PASS and the printed max rel diff ~1e-6..1e-5.
3. Kernel time: nsys `--cuda-graph-trace=node` at C160, `MYNAH_CUDA_SHARED_VOICE=1` with and without
   `MYNAH_CUDA_ATTN_SPLIT=1`: `k_self_attention_bf16_batch_split` vs `k_self_attention_bf16_batch_fast` total per
   step. `ncu --kernel-name regex:k_self_attention_bf16_batch_ --metrics
   dram__bytes_read.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,gpu__time_duration.sum`
   on one replay for both: same DRAM bytes expected, higher DRAM throughput % for split.
4. Knee A/B (C160/C192/C208, interleaved): arm A `MYNAH_CUDA_SHARED_VOICE=1`, arm B A + `MYNAH_CUDA_ATTN_SPLIT=1`;
   also with `MYNAH_CUDA_BF16_FUSE=1` (BF16OUT path).
5. Audio: not bit-identical to the fast kernel (summation order). Seeded runs must be bit-identical between two
   runs of arm B and across batch compositions (same request alone vs inside C24), which is the determinism claim.

Risks: (a) register pressure/spills from the 8 `uint4` in flight (fallback: UNROLL 2); (b) short rows (n < 48)
leave warps idle, harmless at the 130-700 positions of the server; (c) four warps per (row, head) cap the
parallelism of a single long row (n in the thousands, one row) at ~1 block per head: fine for batched decode, the
single-row step does not use this kernel; (d) TF32-class numerics: the change is in the last bits only, but it is
a different reduction order from the fast kernel, so any golden-md5 test must be run with the flag off.

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
  `--pocket-self-check` compares a text pushed in pieces vs whole bit for bit (memcmp), and the gang decoder vs the
  solo decoder within 1e-4 — the gang transposed conv is a GEMM, the solo one SIMT, so that pair was never
  bit-equal): PASS with `DECODER_FUSE`, with `SHARED_VOICE`, with `PREFILL_BF16TC`, and with all three. Each flag's
  start-up line appears, so the new paths really ran. The BF16 prefill tile is batch-invariant (bitwise). The fused
  decoder's exactness comes from the temperature-0 runs: off vs fuse, same build, 24 requests all 99 dB (identical).
- Under concurrent serving load the default configuration is not batch-invariant (TF32 cuBLAS decode GEMMs pick
  their algorithm by width): the same 24 seeded requests at concurrency 24 and 3 differ in all 24 md5s, flags off.
  So md5 comparisons between two server runs only hold when the gang history is the same; at temperature 0 two runs
  of one build are usually identical (99 dB) but one request in 24 can diverge (24 dB) from timing alone.

## Work items (2026-10-02)
1. [running] knee: shared voice + decoder fuse + bf16 prefill tile vs cuBLAS prefill at C192/C208/C224.
2. [running] knee: all-bf16 weights (MYNAH_CUDA_QUANT=bf16 + PREFILL_BF16TC) vs fp32 weights; listening pack for
   the stages a reference PyTorch engine kept fp32 (flow head, Mimi transformer).
3. [branch pocket-cuda-shv2] shared voice phase 2: rows without the voice prefix (VRAM for C224+; C224 OOMs today at
   22.5 GB during KV growth).
4. [branch pocket-cuda-bf16dec] bf16 decode linears: per-stage switch, LN/GELU writing bf16, bias folded.
5. [branch pocket-cuda-decfuse] decoder fusion part 2: bias, residual, convtr overlap/fold/copy.
6. [analysis] .work/pocket-cuda-inefficiencies-and-l40s.md: syncs, copies, allocations, threads, FP8, L40S knobs.
7. Then: measured wins default ON for Pocket (env flags kept to switch off), C208 2-min + 10-min soaks, C224+,
   docs/cuda-serving.md update, PR, merge, remove worktrees.
8. Later (future option): an L40S (same sm_89 family, 142 SMs, ~2.9x bandwidth, 48 GB) 1-2 h knee session.

## bf16 decode linears

Branch `pocket-cuda-bf16dec` from 0dce18b. Goal: bf16 FlowLM weights as a clear win on the batched decode step, in
the scope the reference PyTorch engine used (FlowLM transformer Linears bf16; residual stream, norms, flow head, EOS
head and codec fp32; there it gave -8/9 % RTF p95 and +11 % throughput). Nothing changes with the new flags unset:
the f32/TF32 path issues the same kernels with the same arguments, and the start-up line is unchanged.

### `MYNAH_CUDA_QUANT_STAGES` (default all)
Comma list of `backbone`, `flow`, `mimi` (or `all` / `none`): the resident stages that take the BF16 weight copies
under `MYNAH_CUDA_QUANT=bf16`. Unset means all three, which is what `MYNAH_CUDA_QUANT=bf16` always did. An unknown
name fails the model load. Implementation: `state->cuda_bf16_stages`, consulted by `pocket_cuda_f32_qtype(state,
stage)` (backbone for `pocket_cuda_tar_qtype`, flow for `pocket_cuda_flow_qtype`, mimi for
`pocket_cuda_codec_qtype`), so every consumer (batched step, single-row step, BF16 prefill tile, flow head, Mimi
tile and fallback) follows. `pocket_cuda_stage_encoding` uses the same mapping, so the start-up line reports it:

    mynah-tts: CUDA quant=bf16 backbone=bf16 flow=f32 mimi=f32 kv=bf16 bf16_stages=backbone bf16_fuse=on

(`bf16_fuse=on` only when the backbone really is bf16, the flag is set and the backend has the fused ops.) The
existing per-group mechanism (`MYNAH_QUANT_GROUPS`) was not reused: it selects CPU qmat encodings, and a bf16 group
there is rejected by the resident path (bf16 x f32 on the CPU is a different representation from the device's bf16 x
bf16 tensor-core GEMM). `MYNAH_QUANT_GROUPS` still decides int8; the new variable only narrows where the f32 groups
upgrade to the BF16 copies. The EOS head and `input_linear` were never part of `MYNAH_CUDA_QUANT=bf16`.

### `MYNAH_CUDA_BF16_FUSE` (default 0 while measured)
In `pocket_cuda_backbone_step_batch`, a layer whose four Linears are all BF16 runs
`pocket_cuda_backbone_layer_bf16_fused`. The backend's BF16 activation buffer (the one
`mynah_cuda_matmul_bf16_d2d` casts into, already reserved before any capture by `mynah_backend_bf16_reserve`) becomes
a staged GEMM operand that the producers write directly:

| step | unfused BF16 (today) | fused |
|---|---|---|
| LN1 | `k_layer_norm` -> f32, `k_f32_to_bf16` | `k_layer_norm_bf16` (same body, BF16 store) |
| QKV | GEMM, `k_bias_add` | GEMM from staged, no bias |
| RoPE | `k_rope_qk_batch` | `k_rope_qk_bias_batch` (q/k bias before rotation, v bias) |
| attention | `k_self_attention_bf16_batch_fast` -> f32 | same kernel, `BF16OUT` template store into staged |
| out_proj | `k_f32_to_bf16`, GEMM, `k_bias_add` | GEMM from staged, no bias |
| residual | `k_residual_add` | `k_residual_bias_add` (out_proj bias) |
| LN2 | `k_layer_norm`, `k_f32_to_bf16` | `k_layer_norm_bf16` |
| FFN1 | GEMM, `k_bias_add` | GEMM from staged, no bias |
| GELU | `k_gelu` in place, `k_f32_to_bf16` | `k_bias_gelu_bf16` (bias + GELU, BF16 store) |
| FFN2 | GEMM, `k_bias_add` | GEMM from staged, no bias |
| residual | `k_residual_add` | `k_residual_bias_add` (FFN2 bias) |

Kernels per layer (batched decode, BF16 KV, fast attention, counting each cuBLAS GEMM as one): f32/TF32 15, BF16
unfused 19, BF16 fused 11. Per 24-layer step: 456 -> 264 nodes (-192), against 360 for the TF32 default. The graph's
memcpy nodes (RoPE positions, attention tables) are unchanged. HBM traffic removed per layer at 200 rows is ~25-30 MB
(the f32 round trips of LN output, QKV bias, attention output, FFN1 bias/GELU/cast and the two biases), ~0.6-0.7 GB
per step, i.e. ~2 ms at the L4's bandwidth plus the launch gaps of 192 nodes. On top of that the BF16 GEMMs read half
the weight bytes of TF32 (backbone ~1.15 GB -> ~0.58 GB per step) on twice the tensor throughput, so the backbone
GEMM share (~7 ms at C160) should roughly halve. Expected: a few ms per step out of ~64 ms at C160, i.e. the same
order as the reference's -8 %; to be measured.

Numerics: every fused kernel computes the same FP32 expression in the same order as the sequence it replaces (the
intermediate stays in a register instead of a global round trip; no new multiply-add is exposed to contraction), the
BF16 rounding is the same RNE helper, and the staged GEMM is the identical `cublasGemmEx` call (same pointer, shape,
types, `CUBLAS_GEMM_DEFAULT`). So the audio should be bit-identical to `MYNAH_CUDA_QUANT=bf16` with the flag off.
`--gpu-self-test` now checks this bit for bit (memcmp) for LN, bias+GELU, QKV bias+RoPE and bias+residual
(`cuda_bf16_fuse_self_test`). The attention BF16 store is the cast of the very same value; if the fast kernel cannot
take a call, the legacy kernel writes f32 and a cast stages it. With f32 KV the attention keeps its cast (inside
`mynah_backend_matmul_bf16_d2d`, bias NULL): 12 kernels. A layer with any non-BF16 Linear (int8 group) takes the
unfused sequence. Not fused: the single-row step (`count < 2`) and the prefill tile (already `k_tile_gemm_bf16tc`).
Graph safety: no allocation in the sequence; the bias device copies are cached on the first, uncaptured step (the
graph is only captured after one successful uncaptured step) and are the same cache entries the GEMM epilogue used.
cuBLASLt (`CUBLASLT_EPILOGUE_BIAS`) was not used: the build links only `-lcublas`, and folding the bias into the
consumer kernel removes the epilogue pass without a new dependency.

### What to measure on the L4
Correctness first, on the build of this branch:
- `--gpu-self-test` PASS (includes the bitwise fused-kernel check).
- Start-up line of each arm below shows the expected `backbone=/flow=/mimi=` and `bf16_fuse=on` only in the fused arms.
- Bit identity fused vs unfused: the same 24 seeded requests, temperature 0, sent at once at concurrency 24, twice per
  arm (C and D below); md5 equal between C and D wherever the two runs of C agree with each other (the gang history
  caveat of the correctness section applies: cuBLAS picks its algorithm by width, so only equal histories compare).
- Quality of the bf16 backbone itself (arm C vs A) is a model change, not a kernel one: ASR WER on the usual texts and
  a listening pass before it can become a default.

Knee A/B (60-s closed loop, 4 voices, `--max-batch 208`, two interleaved rounds, C160/C192/C208), all arms on top
of the current best `MYNAH_CUDA_SHARED_VOICE=1`:

| arm | env |
|---|---|
| A | `MYNAH_CUDA_SHARED_VOICE=1` |
| B | A + `MYNAH_CUDA_QUANT=bf16` (all stages, unfused) |
| C | A + `MYNAH_CUDA_QUANT=bf16 MYNAH_CUDA_QUANT_STAGES=backbone` |
| D | C + `MYNAH_CUDA_BF16_FUSE=1` |
| E | B + `MYNAH_CUDA_BF16_FUSE=1` |
| F | D + `MYNAH_CUDA_PREFILL_BF16TC=1` |

Read: C-A is the bf16 backbone at today's kernel cost, D-C the fusion alone (expect step time down, audio identical),
E-D whether bf16 flow/Mimi add anything on top, F the combination with the BF16 prefill tile. One nsys capture of D at
C160 (`--cuda-graph-trace=node`) should show per layer 4 bf16 GEMMs, `k_layer_norm_bf16` x2, `k_rope_qk_bias_batch`,
`k_bias_gelu_bf16`, `k_residual_bias_add` x2 and the attention, and no `k_bias_add`/`k_f32_to_bf16` in
`step.backbone`.

## One sync per frame (`MYNAH_CUDA_ONE_SYNC`, default 0)

Branch `pocket-cuda-onesync` from e4b465e. Target: action 1 of `.work/pocket-cuda-inefficiencies-and-l40s.md`
(the measured ~7 `mynah_backend_sync` per step). Scope done here: the four syncs on the critical path between the
condition projection and the flow head; the PCM sync of the decoder gang stays (see "Not done").

### Sync map (steady state, batched decode, Mimi tile + decoder gang defaults; line numbers at this commit)

| # | stage | flag off | flag on |
|---|---|---|---|
| 1 | condition (`input_linear`) | `pocket_cuda_condition_batch`, sync `src/engine_pocket.c:1988` (+ 4 KB/row D2H, host finite scan) | queued, no D2H (`pocket_onesync_step`, `:8034`) |
| 2 | backbone | `pocket_cuda_backbone_step_batch_impl`, sync `:7470`, host commit | same graph, queued with `defer = 1`; host commit after the frame sync (`pocket_cuda_backbone_step_commit`, `:7005`) |
| 3 | EOS logits | `pocket_cuda_eos_batch`, sync `:2047` | queued linear + 1 float/row D2H into pinned memory |
| 4 | flow head | `pocket_cuda_flow_step_batch`, hidden re-uploaded from the host, sync `:7676` | `pocket_cuda_onesync_flow_queue` (`:7835`): cond gathered from `cuda_norm` on the device, own graph per width |
| 5 | noise + LSD add | host, `pocket_emit_batch` | noise: host, drawn in step; add: device (`k_residual_add`), latent D2H |
|   | **the one sync** | | `pocket_onesync_step`, `:8084` |
| 6 | PCM handoff | `pocket_decode_audio_batch`, sync `:11723` | unchanged |

Other syncs that exist with either setting and are not per frame: KV growth (`:6554`), prefill/admission, the
non-tile codec transformer (`:11326`, only with `MYNAH_CUDA_MIMI_TILE=0`), fallbacks. So per batched frame:
5 -> 2 counted syncs, i.e. **3 fewer per frame**; the measured 6.94 syncs/step should drop by ~3 at equal load (the
rest is admission/prefill/growth).

### What runs where with the flag on
`pocket_step_batch` (`:9866`) tries `pocket_onesync_step` first when the scratch has the chain and every row steps
(batch >= 2, no budget-exhausted row, no `noise_fn`, resident CUDA backbone + flow, COND_IN/COND_EOS groups resident):
1. host: previous latents -> pinned buffer; each row's noise for this step drawn (RNG state saved first).
2. queue: H2D latents, `input_linear` -> `cuda_x` (the backbone's condition-input graph key, unchanged graph).
3. queue: backbone (graph replay/capture exactly as before; hidden D2H and the optional host K/V mirror D2H are
   graph nodes as before).
4. queue: EOS linear on `cuda_norm` (count rows, same call as before) + D2H.
5. queue (graph key `0x500000 + width`): noise H2D, time H2D, `gather_rows_to_batch` cond (rows >= count repeat row 0,
   as the ordinary flow does), `flow_batch_dev`, latent = copy(noise) + flow, D2H flow output and latent.
6. one `mynah_backend_sync`; then the backbone commit (finite gate, hidden rows, offsets, valid flags), EOS logits into
   each ctx, flow finite flag. `pocket_emit_batch` then only decides EOS, keeps or undoes the early draw, and copies
   flow output and latent.

### Why sampling and the audio stay identical
- Noise: same RNG, same expression (`noise_std * pocket_rng_normal(ctx)`), same per-row order; only the moment moves
  (before the chain instead of after the EOS logit). The RNG has no other consumer between step and emit. The one case
  where the ordinary emit draws nothing is a terminal step (`step >= eos_step + frames_after_eos`, or a budget row):
  there emit restores the saved RNG state (`pocket_onesync_rng_restore`), so the next segment's draws are unchanged.
  A draw no emit consumed (an emit that never ran) is undone at the next step; `reset` clears it.
- Condition, backbone, EOS: the same calls on the same buffers and widths (same graph keys); only the syncs between
  them are gone.
- Flow: the cond rows are the same bytes (device copy of `cuda_norm` instead of the host round trip of the same
  floats), the noise is the same floats, the width is `exec(count)`, the buffers and kernels are the ordinary ones.
  That equals the ordinary call only when every row takes a latent. In a step where some row ends, emit ignores the
  chained result and runs the ordinary `pocket_cuda_flow_step_batch` on the subset (one extra sync, once per request
  or segment end), so that step is the ordinary one too.
- Latent: `k_residual_add` computes `noise + flow` as one fp32 addition (no multiply, so no FMA contraction; no fast
  math), bit-equal to the host `noise[d] + flow_out[d]`.
- Flag off: no buffers are allocated, `cuda_onesync_enabled` stays 0, and every changed line on the ordinary path is
  either a no-op test of a zero field or the backbone tail moved verbatim into `pocket_cuda_backbone_step_commit`.

### Failure behaviour
- Not eligible this step (backbone returns 1, e.g. a row off the device): nothing committed, early draws undone, the
  ordinary path runs (the queued projection is simply redone on the same stream).
- Non-finite hidden/flow row, or an offset that cannot advance: nothing committed, the ordinary path redoes the frame
  (the backbone rewrites the same K/V slot with the same values) and reports it as before; a non-finite chained flow
  makes emit run the ordinary flow.
- Launch or sync error: drain, forget the scratch's graphs, disable the chain for this scratch (stderr line
  "MYNAH_CUDA_ONE_SYNC disabled after a failure"), continue on the per-stage path.

### How to verify on the L4
1. Start-up line: `MYNAH_CUDA_ONE_SYNC: condition, backbone, EOS and flow head share one stream sync per batched
   frame`; no "disabled after a failure" line during the runs.
2. Sync count: under a fixed closed loop (e.g. C64 for 60 s), `mynah_backend_sync_calls_total` delta divided by the
   step delta (`/health` steps), flag off vs on: expect about -3 per step. nsys: one `cudaStreamSynchronize` between
   `step.backbone` start and the decoder, not four.
3. Bit identity: the same 24 seeded requests, temperature 0, SEANet fp32, sent at once at a fixed concurrency to a
   fresh server, flag off vs on, same build: md5 per request identical (the gang-history caveat of the correctness
   section applies; compare off vs off first). Include requests long enough to end at different steps (terminal-step
   subset flow), multi-segment requests (RNG continues across the segment boundary) and `frames_after_eos` 0.
4. Stronger, per tensor: `MYNAH_POCKET_DUMP=<dir>` on both arms; `hidden`, `eos`, `flow_out`, `latent` .npy files
   byte-identical.
5. `--pocket-self-check` with the flag on (pedantic `MYNAH_CUDA_TF32=0 MYNAH_CUDA_SEANET_BF16=0`): PASS, including the
   latent-NaN and KV-NaN atomicity cases (they now go through the chain, fall back, and must still refuse atomically).
   `--gpu-self-test` PASS. Also `MYNAH_CUDA_GRAPHS=0` (uncaptured chain) and `MYNAH_CUDA_WIDTH_BUCKETS=0`.
6. Knee A/B at C192/C208 on top of the best flag set: step time and scheduler CPU.

### Risks
- One extra graph per width bucket (15 by default; cap 384).
- The chain runs the flow head on rows whose step turns out terminal, then reruns the ordinary flow on the subset:
  one extra flow pass and sync per request end (rare).
- The hidden rows still come back to the host (4 KB/row, in the same sync): needed for the finite gate, the dump and
  the CPU fallbacks. A device NaN flag could replace the gate later.
- `gather_rows_to_batch` uses the backend's shared metadata buffer (`dev_batch_k_cache`), which the backbone's
  attention tables also use; this is correct because everything is ordered on one stream.
- Rows with a host `noise_fn` (parity harness) never take the chain.

### Not done (the rest of action 1)
- PCM sync (#6): the decoder gang is a separate engine call (`decode_audio_batch`) issued by the driver after
  `emit_batch`, and the codec input is the host latent (`pocket_cuda_codec_gang_upsample` uploads `denorm`). Merging it
  into the frame sync needs (a) the denormalisation, upsample, Mimi tile and SEANet fed from the device latent, and
  (b) knowing before the sync whether frame N exists, because the codec state cannot be rewound. (b) is decidable on
  the host before the step whenever `frames_after_eos >= 1`: a row terminates at step N only if its `eos_step` was set
  at an earlier step; only `frames_after_eos == 0` makes it depend on the logit of step N. So a frame-major chain is
  possible for the default config, but it changes the driver/engine contract (step, emit and decode in one call) and
  the delivery logic in `src/inference.c` (`stream_gang`).
- PCM in int16 on the device, the hidden D2H, and the EOS/latent/flow/hidden copies packed into one D2H.

### 2026-10-02 13:20 CEST — merged build (all branches), knee screen, max-batch 240
Stream RTF p95 (60 s closed loop, 4 voices, one round; 0 stalls, 0 failures everywhere):

| arm | C208 | C224 | C240 | audio-s/s @C240 | peak VRAM |
|---|---|---|---|---|---|
| A: SHARED_VOICE (rows stripped) + PREFILL_FIXED=0 + DECODER_FUSE (parts 1+2) | 0.705 | 0.745 | 0.787 | 250 | 20.8 GB |
| B: A + ATTN_SPLIT | 0.674 | 0.711 | 0.749 | 267 | 20.5 GB |
| C: B + QUANT=bf16 QUANT_STAGES=backbone BF16_FUSE | 0.768 | 0.811 | 0.854 | 235 | 19.6 GB |

- Same day, before the strip and fusion part 2, SHARED_VOICE + PREFILL_FIXED=0 + DECODER_FUSE (part 1) read 0.835 at
  C208 and C224 failed (OOM during KV growth). The strip removes the per-row prefix (start-up line "backbone KV rows
  do not store the 126-position voice prefix") and C224/C240 now run.
- C was slower because with bf16 weights every bf16 tile GEMM took the SIMT kernel, losing the cuBLAS prefill:
  fixed in d79971c (cuBLAS bf16 for tiles that do not ask for a fixed order); re-test queued.
- Temperature-0 identity (same build, SEANet fp32, 24 requests): strip and fusion part 2 median 99 dB (identical;
  a few requests diverge from gang timing alone); ATTN_SPLIT changes the summation order (self-test max rel diff
  5e-7 vs the fast kernel), so trajectories diverge at temperature 0 (median 24 dB), the same class as TF32 vs fp32.

## KV growth in place (VMM) (`MYNAH_CUDA_KV_VMM`, default 0)

Branch `pocket-cuda-kvvmm` from 500479b. Target: the C272 failure ("the CUDA backbone step failed for a device-owned
request", peak 22.55 GB with `--max-batch 288`). `.work/pocket-cuda-inefficiencies-and-l40s.md` (1.3): the KV growth
of `pocket_cuda_backbone_reserve` does `cudaMalloc(new)` + 2 x 24 D2D copies + a stream sync + `cudaFree(old)` inside
the step pre-flight, so every growth briefly holds old + new (a 512 -> 768 stored-position growth of one 24L BF16 row
holds 48 + 72 MB to end at 72 MB) and stalls the stream twice (sync + the implicit sync of `cudaFree`). With the flag
on, a growable row is a virtual address range whose growth maps more device pages after the mapped ones: no copy, no
sync of ours, no second allocation, the base pointer never moves. Off: no driver call is made, every allocation, layout,
pointer table and kernel argument value is the old one.

### Design
- **API, no `-lcuda`.** `cuMemGetAllocationGranularity`, `cuMemAddressReserve/Free`, `cuMemCreate/Release`, `cuMemMap/
  Unmap`, `cuMemSetAccess`, `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED)` are resolved
  at run time through the runtime (`cudaGetDriverEntryPointByVersion(..., 12000, ...)` from CUDA 12.5,
  `cudaGetDriverEntryPoint` before); only the driver TYPES come from `<cuda.h>`. Reason: the compile-only CI jobs
  (sm_70/89/90 on 12.6, sm_120 on 12.8) run the built binary in a container without a driver; a hard `-lcuda` would
  need the stub library at link time and leave a binary that does not load where `libcuda.so.1` is missing. The link
  line is unchanged (`-lcublas` + the runtime nvcc adds). Checked with clang `-x cuda` against the 12.4 and 12.6
  runtime headers (host + sm_89); not compiled with nvcc here.
- **Probe once, at model load** (`mynah_backend_kv_vmm_probe`): entry points present, VMM attribute set, granularity
  (`CU_MEM_ALLOC_GRANULARITY_MINIMUM`, 2 MiB on the L4). Physical memory is `CU_MEM_ALLOCATION_TYPE_PINNED` at
  `CU_MEM_LOCATION_TYPE_DEVICE` (VRAM; never host memory), read-write access for the device. Start-up line:
  `mynah-tts: MYNAH_CUDA_KV_VMM=1: backbone KV rows of the prefill tile path are position-major VMM ranges that grow in
  place (granularity 2048 KiB, growth chunk 256 positions, reservation in units of 1024 positions)`. Otherwise one line
  `mynah-tts: warning: MYNAH_CUDA_KV_VMM=1 ignored (<reason>); backbone KV growth keeps cudaMalloc + copy`, also when
  `MYNAH_CUDA_RESIDENT`, `MYNAH_CUDA_KV_GROW` or `MYNAH_CUDA_PREFILL_TILE` is off.
- **Which rows.** `pocket_cuda_kv_vmm_planned`: the probe passed and `pocket_cuda_kv_grow_expected(ctx)` (the
  prefill-tile, device-owned rows that MYNAH_CUDA_KV_GROW already makes growable). Everything else keeps the plain
  cache. A VMM allocation that fails for any reason logs once and leaves that context on the plain cache
  (`cuda_kv_vmm_refused`).
- **Layout (position-major).** Row bytes `[stored position][layer][K|V][attn]`: stored position s is absolute s + S
  (S = `cuda_backbone_kv_skip`, the shared-voice strip). One position = layers x 2 x attn elements = 24 x 2 x 1024 x 2 B
  = 96 KiB (BF16), 21.3 positions per 2 MiB page; mapping per K/V plane instead would have wasted up to a page per plane
  (48 per row). Expressed with the existing parameters: position stride = `pocket_cuda_kv_stride(ctx)` = layers x 2 x
  attn (= 49152 elements; attn for plain rows), layer plane base = row + (l x 2 + plane) x attn
  (`pocket_cuda_kv_plane`), position-addressed base = plane - S x stride (`pocket_cuda_kv_position_base`, integer
  arithmetic, never dereferenced below S, exactly the phase-2 contract with the stride in place of attn). Alignment
  for the fast/split kernels holds (base 2 MiB aligned, plane offsets multiples of 2 KiB, stride % 8 == 0).
- **Reservation.** Per row `round_up(max_seq_len, 1024)` positions x position bytes, `max_seq_len` being the host
  ceiling that `pocket_cuda_backbone_reserve` never passes (voice + text capacity + max_steps + 1, ~1700 with the
  default budget -> 2048 positions = 192 MiB of addresses, no memory). A long-form text that raises the ceiling goes
  through `mynah_engine_pocket_reserve_text`, which releases and re-allocates the cache (new reservation).
- **Allocation** maps `kv_bytes` (the same stored-positions x position-bytes as the plain cache) rounded up to pages;
  the row's capacity is then everything the mapped pages hold (`mapped / position_bytes + S`, capped at the ceiling),
  i.e. the page rounding is headroom, not waste.
- **Growth** (`pocket_cuda_backbone_reserve`, VMM branch): target = old capacity + whole chunks of
  `MYNAH_CUDA_KV_VMM_CHUNK` positions (default 256, the old growth cadence) covering `needed`, mapped size rounded up
  to pages, one `cuMemCreate + cuMemMap + cuMemSetAccess` per growth (one chunk, unmappable on its own later). No
  copy, no `mynah_backend_sync`, no free. `MYNAH_CUDA_KV_GROW_LOG=1` prints "grew A -> B positions in place (VMM, ...)".
- **Release** (`pocket_cuda_backbone_release`, slot destroy): unmap + release every chunk, free the reservation, after
  the existing drain. `mynah_backend_dev_free` on a VMM pointer is routed to the same release (a lookup only while a
  range is live), so no path can `cudaFree` a VMM address.
- **Slot pool** (decision: keep the mappings, trim on reuse). A parked VMM row keeps its mapped pages, like a parked
  grown plain cache. `pocket_cuda_slot_acquire(..., vmm, reserve)`: a VMM request takes only a VMM set whose
  reservation covers its ceiling (tightest mapped size at or above its start, else the largest), a plain request never
  takes a VMM set. `pocket_cuda_backbone_alloc` then resizes in place to [start size, 2 x start size]: maps more if
  short, unmaps whole trailing chunks if above twice the need (the bound the plain path enforces by free +
  reallocate). `cuMemUnmap` is documented as possibly synchronous: only on this admission-time trim of an idle set
  and on release, where the plain path did a `cudaFree` anyway.
- **Graph capture.** No map/unmap inside a capture: every growth caller runs before `batch_begin`/`graph_begin`
  (batched step pre-check `pocket_cuda_backbone_step_batch_impl`, step pre-flight in `pocket_step_batch`, single-row
  step, prefill tile before its staging, upload); the one-sync chain queues the condition projection before the
  backbone pre-check, uncaptured. Growth adds pages after the ones in use, so work already queued on the row is
  unaffected, and the base never moves, so graph-replayed host tables stay valid trivially.

### Every reader/writer of a VMM row
| site | change |
|---|---|
| `k_self_attention_bf16_batch_fast`, `_split`, legacy `k_self_attention_bf16_batch<SHARED>`, f32 `k_self_attention_batch` | none: per-row `cache_strides[i]` already used for the K/V write and every read; the engine now passes `pocket_cuda_kv_stride` (batched step tables, pad rows keep attn) |
| single-row `k_self_attention_bf16<SHARED>`, f32 `k_self_attention` | none in the kernels; `pocket_cuda_backbone_step` passes `kv_stride` instead of `attn_dim` and mirrors the new slot from `position * kv_stride` |
| `k_gather_kv_bf16_batch` / `k_gather_kv_batch` (host mirror) | none: indexes `positions * cache_strides` |
| RoPE (`k_rope_qk_batch`, `k_rope_qk_bias_batch`) | none: touches only qkv; the attention kernels store K/V |
| `k_tile_rope_store`, `k_tile_attention`, `k_tile_attention_grouped` | new nullable `strides` table, per row (pitch, V offset): K slot at `kv + slot * pitch`, V at `kv + voff + slot * pitch`; NULL = (dim, ring x dim), the old arithmetic; `mynah_backend_tile_desc.kv_strides`, staged after the skips only when non-NULL (the plain staging layout is unchanged); validated on the host (no K/V overlap) |
| prefill tile voice copy (`pocket_cuda_prefill_tile`, S = 0 rows) | one `cudaMemcpy2DAsync` per layer and plane (`mynah_backend_copy_dev_bytes_2d`): the voice plane `[P][attn]` lands strided; 48 copies as before |
| prefill tile kv/ring tables | kv = `pocket_cuda_kv_plane`, ring = stored capacity (never wraps), `kv_strides` only if a VMM row is in the call; plain rows in the same call get their plane-major pair |
| shared-voice prefix reads (`MYNAH_CUDA_SHARED_VOICE`, strip) | unchanged: prefix planes come from the device voice cache; a stripped VMM row is biased by S x stride |
| `pocket_cuda_backbone_upload` | refuses a VMM row (device-owned only; unreachable) |
| `pocket_seed_backbone` non-tile path | a VMM row with S = 0 is swapped for a plain cache (`cuda_kv_vmm_refused`) and continues as before; S > 0 keeps the phase-2 behaviour (CPU). Not expected: the plan mirrors the tile's test |
| `pocket_cuda_backbone_reserve` | VMM branch above |
| slot park/acquire/free, release | VMM flags and reservation carried; free by kind |
| padding rows (`cuda_pad_kv`) | unchanged (own plain one-position cache, stride attn); mixed batches of plain and VMM rows share graphs (strides are per-row replayed tables; the fast-kernel eligibility, stride % 8 and 16-byte alignment, is the same for both) |

### Memory accounting
- `device_memory_free_bytes` (`cudaMemGetInfo`) already counts mapped VMM pages as used; the reservation is not memory.
- New backend metrics (appended to `mynah_tts_backend_metrics`, exported on `/metrics`): `mynah_backend_kv_vmm_rows`
  (live + parked VMM rows), `mynah_backend_kv_vmm_mapped_bytes`, `mynah_backend_kv_vmm_reserved_bytes` (gauges),
  `mynah_backend_kv_vmm_maps_total`, `mynah_backend_kv_vmm_unmaps_total` (counters). The `/metrics` buffer went from
  16 KiB to 24 KiB (it was at ~15 KiB).
- `cuda_backbone_kv_bytes` of a VMM row is its mapped size, `cuda_backbone_kv_reserved` its reservation. First VMM row:
  "pocket: first CUDA backbone KV row in a VMM range: N positions mapped (B bytes), R bytes reserved".
- Start-up reservations (slot-pool prefill, width-bucket warm-up, +17.5 GB at 288) are unchanged: a prefilled VMM set
  maps the same start size a plain set allocates. What goes is the growth transient and the stream stalls.

### Self-test (`--gpu-self-test`, always run when the device supports VMM; skipped otherwise)
`cuda_kv_vmm_self_test`: (1) a range keeps its bytes and base across a growth in place (1 -> 3 pages) and a trim back
to 1 page, refuses a resize past its reservation, and is released by `mynah_backend_dev_free`; (2) the fast, split
and legacy decode attention (with shared prefix tables) and the K/V gather give bit-identical results on a
plane-major cache and on the same values position-major in a VMM range (3 layers, layers 0 and 2, positions 5/130/300,
one row stripped by 3 positions and biased below the start of the range, so a stray read below the skip faults).

### To test on the L4 (same build for every A/B)
1. `make cuda cuda-server CUDA_ARCH=sm_89`; `./build/cuda/mynah-tts --gpu-self-test cuda` PASS (with
   `MYNAH_CUDA_KV_VMM=1` a "KV VMM self-test skipped" line would mean the probe failed). Server start with the flag:
   the start-up line, no warning.
2. Pedantic: `MYNAH_CUDA_TF32=0 MYNAH_CUDA_SEANET_BF16=0 MYNAH_CUDA_KV_VMM=1 --pocket-self-check` PASS; again with
   `MYNAH_CUDA_SHARED_VOICE=1` (strip: biased VMM rows) and with `MYNAH_CUDA_KV_GROW_INITIAL_STEPS=8`.
3. Temperature-0 md5 identity, 24 seeded requests (4 voices, SEANet fp32), flag off vs on, same build and the same
   concurrency (gang-history caveat: compare off vs off first). The layout must not change a value. Repeat with
   `MYNAH_CUDA_KV_GROW_INITIAL_STEPS=8 MYNAH_CUDA_KV_GROW_LOG=1` (growth in place mid-request: the log must say
   "in place (VMM"), with `MYNAH_CUDA_SHARED_VOICE=1`, `MYNAH_CUDA_ATTN_SPLIT=1`, `MYNAH_CUDA_BACKBONE_ATTN=legacy`,
   `MYNAH_CUDA_TILE_ATTN_GROUPED=0`, a single request alone (single-row step) and a multi-segment / appended text.
4. `compute-sanitizer --tool memcheck` on a short C4 run with `MYNAH_CUDA_KV_VMM=1 MYNAH_CUDA_SHARED_VOICE=1
   MYNAH_CUDA_KV_GROW_INITIAL_STEPS=8`.
5. Knee with `--max-batch 288`, best flag set, C256/C272/C288, flag off vs on (and `MYNAH_CUDA_KV_VMM_CHUNK=64`):
   failures, stream RTF p95, stalls, `nvidia-smi` peak, `mynah_backend_kv_vmm_mapped_bytes` and maps/unmaps over the
   run. Expect C272 without "the CUDA backbone step failed for a device-owned request".
6. Performance of the layout: nsys `--cuda-graph-trace=node` at C208, attention kernel time per step off vs on, and
   the CUDA API trace of `cuMemCreate/cuMemMap/cuMemSetAccess` (cost of one growth; whether it stalls the stream).

### Risks
- **TLB reach.** Plane-major: one layer's K plane of a row spans n x 2 KiB (one or two 2 MiB pages); position-major:
  the same reads are 96 KiB apart, so one layer's attention touches every page of the row (~15 pages at n ~ 300),
  ~10x more distinct pages per kernel at C208+. DRAM efficiency per access is the same (each read is a full 128-byte
  head row either way), but TLB misses could slow the decode attention. Step 6 decides. If it costs, the alternative is
  a block-major page ([layer][K|V][21 positions][attn] per 2 MiB page), which needs a page-indexed kernel, not strides.
- The driver documents `cuMemMap`/`cuMemSetAccess`/`cuMemUnmap` as possibly synchronous: a growth may still stall the
  device (without the copy, the second allocation and the peak). Measure in step 6; the chunk knob trades growth count
  against mapped headroom.
- f32 KV VMM rows (192 KiB per position) are supported by the same code but untested; the default KV is BF16.
- A VMM row that unexpectedly leaves the tile path is re-allocated plain (S = 0) or goes to the CPU (S > 0), as the
  phase-2 rows already did.

### 2026-10-02 14:35 CEST — bf16 backbone through cuBLASLt (60 s knee, max-batch 256, one round)
All arms: SHARED_VOICE (stripped rows) + PREFILL_FIXED=0 + DECODER_FUSE + ATTN_SPLIT + ONE_SYNC.

| arm | C208 | C240 | C256 | audio-s/s @C256 | TTFA p95 @C208 | peak VRAM |
|---|---|---|---|---|---|---|
| B: fp32 weights (TF32) | 0.676 | 0.748 | 0.788 | 266 | 115 ms | 22.1 GB |
| C: + QUANT=bf16 QUANT_STAGES=backbone BF16_FUSE BF16_LT | 0.607 | 0.679 | 0.714 | 303 | 104 ms | 21.5 GB |
| ALL: + bf16 flow head and Mimi (QUANT_STAGES all) | 0.603 | 0.676 | 0.712 | 305 | 104 ms | 21.4 GB |

- Before BF16_LT the bf16 backbone was 10 % SLOWER than TF32: cublasGemmEx with CUBLAS_GEMM_DEFAULT picked small
  bf16 tiles, and the bf16 prefill tile fell back to the SIMT kernel (nsys: 31 % of GPU time, ~15 ms/step; fixed in
  the tile buffer commit). cuBLASLt with a per-shape heuristic (what PyTorch's bf16 F.linear does) makes bf16 the
  expected win: -10 % RTF p95, +14 % audio-s/s against TF32.
- bf16 flow head + Mimi transformer add nothing measurable on top of the backbone.
- Open: `--pocket-self-check` long-form (text in pieces vs whole) fails on the C config and on B + KV_VMM; being
  isolated (suspect ONE_SYNC, which earlier passing builds did not have).

## Defaults (branch `pocket-cuda-defaults`, 2026-10-02)

The measured wins of this branch are the default for the Pocket CUDA path; every variable is kept and `=0` (or
`MYNAH_CUDA_QUANT=f32`) is the rollback, the same pattern as d9b9c89 (SEANet BF16, slot-pool prefill, width buckets).
Empty values read as unset (default), like the earlier defaults.

| variable | new default | where |
|---|---|---|
| `MYNAH_CUDA_SHARED_VOICE` | on (`_STRIP` stays on) | `pocket_cuda_shared_voice_enabled`, `src/engine_pocket.c` |
| `MYNAH_CUDA_DECODER_FUSE` | on (`_FUSE_BIAS` stays on under it) | `decoder_fuse_enabled`, `gpu/cuda/backend_cuda.cu` |
| `MYNAH_CUDA_ATTN_SPLIT` | on | `cuda_backbone_attn_split_enabled`, `gpu/cuda/backend_cuda.cu` |
| `MYNAH_CUDA_ONE_SYNC` | on | `pocket_cuda_one_sync_enabled`, `src/engine_pocket.c` |
| `MYNAH_CUDA_QUANT` | bf16 (Pocket CUDA engine only) | `pocket_cuda_quant_default`, `src/engine_pocket.c` |
| `MYNAH_CUDA_QUANT_STAGES` | `backbone` (was all) | `pocket_cuda_quant_stages_parse`, `src/engine_pocket.c` |
| `MYNAH_CUDA_BF16_FUSE` | on | `pocket_cuda_bf16_fuse_enabled`, `src/engine_pocket.c` |
| `MYNAH_CUDA_BF16_LT` | on | `cuda_lt_init`, `gpu/cuda/backend_cuda.cu` |
| `MYNAH_CUDA_PREFILL_BF16TC` | on for bf16-weight tiles | `cuda_prefill_bf16tc_level`, `gpu/cuda/backend_cuda.cu` |
| `MYNAH_CUDA_PREFILL_FIXED` | 1 (unchanged; one-line switch `POCKET_CUDA_PREFILL_FIXED_DEFAULT`) | `src/engine_pocket.c` |
| `MYNAH_CUDA_KV_VMM`, `MYNAH_CUDA_TILE_TC` | 0 (unchanged) | |

Semantics decided here:
- `MYNAH_CUDA_QUANT` unset (or empty) means bf16 only in the Pocket engine's CUDA state. The shared
  `mynah_cuda_quant_from_env` still reads unset as f32 (its other caller, the int8 qmat request in `src/mynah_tts.c`,
  only tests for int8), so no other model or backend changes. An explicit `MYNAH_CUDA_Q8` (expert int8 recipe) keeps
  the f32 base. `MYNAH_CUDA_QUANT=f32` restores fp32/TF32 weights everywhere, prefill included (see BF16TC below).
  An explicit `MYNAH_CUDA_QUANT=bf16` keeps its meaning except the stages default (`MYNAH_CUDA_QUANT_STAGES=all` for
  every stage).
- `MYNAH_QUANT_GROUPS=none` (the serving profile) does not turn the bf16 default off: groups select int8, and the f32
  groups (`none` = all) are exactly what bf16 upgrades. Nothing changed there; checked in `pocket_cuda_tar_qtype`.
- `MYNAH_CUDA_BF16_LT`: the cuBLASLt handle and its 32 MiB workspace are created by the first bf16 workspace reserve
  (`mynah_cuda_bf16_reserve`, before any capture) instead of at backend open, so an f32 run or another model on a
  CUDA backend never allocates it or prints its line.
- `MYNAH_CUDA_PREFILL_BF16TC`: unset = the bf16 tensor-core tile for tiles whose weights are bf16 and that ask for a
  fixed order (or have no cuBLAS bf16 path); an explicit non-zero value also covers the fixed-order f32-weight prefill
  (the opt-in measured in the first knee); `=0` never. A bf16 tile that does not ask for a fixed order (Mimi tile with
  `QUANT_STAGES=all`, or `MYNAH_CUDA_PREFILL_FIXED=0`) now takes cuBLAS bf16 even with BF16TC on, so
  `MYNAH_CUDA_PREFILL_FIXED=0` really selects the cuBLAS prefill. Switching the default to the cuBLAS prefill is the
  one line `#define POCKET_CUDA_PREFILL_FIXED_DEFAULT 1` -> `0` in `src/engine_pocket.c`.
- Batch invariance: the backend reads `MYNAH_CUDA_QUANT` itself and only knows an explicit bf16, so the self-check now
  asks `pocket_batch_invariant` (backend invariant and no bf16 backbone weights). Pedantic, bitwise setup:
  `MYNAH_CUDA_TF32=0 MYNAH_CUDA_SEANET_BF16=0 MYNAH_CUDA_QUANT=f32` (+ `MYNAH_CUDA_ATTN_SPLIT=0` for the old
  attention summation order, i.e. the numerics of the earlier goldens). Shared voice, decoder fusion, one sync and the
  bf16 fusion are bit-identical and need no switch.

Start-up lines (unchanged text, now printed by default): "decode attention reads voice prefixes ...", "backbone KV rows
do not store the N-position voice prefix", "split (flash-decoding) kernel", "MYNAH_CUDA_ONE_SYNC: ... one stream sync
per batched frame", the decoder fusion line, "bf16 Linears through cuBLASLt", "fixed-order prefill tile on bf16 tensor
cores" (now on first use), and `CUDA quant=bf16 backbone=bf16 flow=f32 mimi=f32 kv=bf16 bf16_stages=backbone
bf16_fuse=on`.

Docs and profile: `docs/cuda-serving.md` (defaults/opt-in tables, pedantic mode, C256 screening level, start-up
lines), `configs/perf/l4-24g-pocket-en-24l-cuda.json` (max-batch/inflight 256, new nulls, C208/C256 screens in
`ceiling` as INCONCLUSIVE until the soaks), 6L profile `MYNAH_CUDA_QUANT` note, `tests/test_perf_profile.py`.

To verify on the L4 before merging (not compiled here, no nvcc):
1. `--gpu-self-test` PASS; server start shows every line above and no "unavailable"/"disabled" line.
2. `--pocket-self-check` with defaults (tolerance mode) and pedantic (bitwise) PASS. The long-form failure noted at
   14:35 on the C config is still open and would now fail the default self-check (and `tools/gpu/provision.sh`).
3. One knee with no flag exported (defaults only) at C208/C256 must reproduce the C arm (0.607 / 0.714); the 14:35
   C arm ran `PREFILL_FIXED=0` (cuBLAS prefill), the defaults run the bf16 tensor-core fixed-order prefill.
4. With every flag `=0` and `MYNAH_CUDA_QUANT=f32`, md5-identical to main 5ae4fd2 defaults (same seeded requests).
