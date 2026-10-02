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
