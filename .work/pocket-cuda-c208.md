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
