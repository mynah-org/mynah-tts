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
