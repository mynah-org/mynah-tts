# Pocket CUDA: SEANet decoder convolutions in bf16 (`MYNAH_CUDA_SEANET_BF16`)

Status: code ready, **not built with nvcc and not run on a GPU** (2026-10-01). Branch `pocket-cuda-seanet-bf16`
from `origin/main` (3934122, v1.5.0). Changes are uncommitted.

Origin: item 1 of "Ports from the Python engine" in `.work/pocket-cuda-cold-burst.md` (branch
`pocket-cuda-cold-burst`). On a reference PyTorch engine of the same 24L model on the same L4, bf16 SEANet convs
(bf16 activations and weights, fp32 accumulation) cut the batched step by 17 % at 192 rows, with SNR 43-45 dB
against fp32 at temperature 0 and no audible difference. SEANet was ~49 % of that engine's step at 160 rows.

## How the CUDA SEANet decoder runs today (v1.5.0)

All of it is in `gpu/cuda/backend_cuda.cu`, section "Resident Pocket SEANet decoder". No cuDNN and no cuBLASLt.

- **Topology:** `decoder_build` → `decoder_add_op`, one op list per request decoder.
  - first conv → for each ratio `[6,5,4]`: ELU + ConvTranspose (kernel 2·ratio), then residual blocks
    (ELU → conv k3 dilated → ELU → conv 1x1 → residual add) → ELU + last conv.
  - 16 positions in per 80 ms frame, 1920 samples out.
- **Conv1d = custom im2col kernels + cuBLAS GEMM.**
  - The causal window and the columns come from `k_decoder_causal_window[_batch]` and
    `k_decoder_causal_columns[_batch]`, written to a per-decoder fp32 `columns` buffer.
  - Bias: `k_decoder_bias_batch` writes it first, and the GEMM runs with beta = 1.
  - The gang (the serving path) uses `cublasSgemmBatched`, with one pointer per request and the shared weight. The
    solo path uses `cublasGemmEx`.
  - `MYNAH_CUDA_DECODER_ONEGEMM=1` (off; measured slower) is the one-GEMM alternative: `cublasSgemm` over the
    concatenated columns, followed by `k_decoder_scatter_bias`.
- **ConvTranspose.**
  - Gang (`MYNAH_CUDA_DECODER_CONVTR_GEMM`, on by default): `k_decoder_convtr_gather` builds X [in][batch·len],
    then one `cublasGemmEx` computes Y = W·X for the whole gang, and `k_decoder_convtr_overlap` overlap-adds the
    taps and the bias.
  - Solo path and `groups != 1`: SIMT fp32 kernels (`k_conv_transpose`, `k_decoder_convtr_batch`).
- **Precision.**
  - All storage is fp32.
  - The default `MYNAH_CUDA_TF32=1` sets `CUBLAS_TF32_TENSOR_OP_MATH` and `CUBLAS_COMPUTE_32F_FAST_TF32`, so the
    GEMMs already run on TF32 tensor cores.
  - `MYNAH_CUDA_TF32=0` switches to pedantic fp32.
- **Glue:**
  - fp32 elementwise kernels driven by per-request pointer tables: ELU, bias, residual, window/tail copy,
    and convtr fold/partial;
  - the whole gang is captured in per-width decoder CUDA graphs.
- **Streaming state:** each request owns its fp32 state, in `op->previous` (the conv left context) and
  `op->partial` (the convtr overlap tail).

## Design (least invasive: same kernels, same GEMMs, bf16 operands)

- **Per-op bf16 weight copy.** `cuda_decoder_op::weight_bf16` is filled in `decoder_add_op` at decoder open
  through the existing backend cache `cached_weight_bf16`. The copy is made once per weight per process and
  shared by every request decoder.
  - It is made for every conv1d and for every convtr with `groups == 1`.
  - It is never made during a capture.
  - The fp32 copy stays, because the fallback kernels still use it.
- **The operand builders write bf16 directly**, so there is no extra cast pass.
  - The im2col kernels and the convtr gather became templates on the output type.
  - A `decoder_put` overload stores either the plain float (bit-identical to before) or the RNE bf16
    (`cuda_bf16_from_float`, the helper the bf16 KV cache uses).
  - The bf16 operands reuse the existing float-sized buffers (half of each is used), so no new allocation is
    needed.
- **GEMMs.** They use the same shapes, layouts, bias prefill and beta. A and B are `CUDA_R_16BF`, C is
  `CUDA_R_32F`, with `CUBLAS_COMPUTE_32F` and `CUBLAS_GEMM_DEFAULT`:
  - gang conv1d: `cublasGemmBatchedEx`;
  - solo conv1d, gang convtr and one-GEMM conv1d: `cublasGemmEx`.
- **What stays fp32:** accumulation, the GEMM output, causal states, bias, ELU, residual adds, the convtr overlap
  and fold, and the returned audio.
- **Solo convtr under the flag** switches from the SIMT kernel to the gang's GEMM form at batch 1.
  - It converts the input to bf16 (`k_f32_to_bf16`, an identity layout at batch 1), then runs the GEMM, then
    the new `k_decoder_convtr_overlap_one`.
  - This way a solo decode rounds like its gang, and the self-check's solo-vs-gang comparison stays inside the
    tensor-core band.
- **`cuda_bf16_math_scope`.** When the handle is in `CUBLAS_PEDANTIC_MATH` (`MYNAH_CUDA_TF32=0`), it lifts the
  pedantic mode for the bf16 call and restores it afterwards. In the default TF32 mode it does nothing.
- **Not touched:**
  - the quantizer upsample (depthwise convtr, a separate path);
  - the codec transformer and the backbone/flow;
  - the CPU SEANet;
  - the non-resident `mynah_cuda_conv1d*` helpers;
  - grouped convtr.

## Flag

- `MYNAH_CUDA_SEANET_BF16=1` turns it on. Unset or `0` leaves it off (the default), following the
  `cuda_env_enabled` convention. It is read once per process.
- **Flag off:** `weight_bf16` stays null, every new branch is skipped, and the float template instantiations do
  exactly the old stores. The code path and the output are the same as today.
- **Flag on:**
  - stderr prints one line at the first decoder open: `mynah-tts: CUDA SEANet decoder convolutions in bf16 ...`;
  - the `bf16_weight_bytes` metric also counts the SEANet copies (a few MB).

## Files touched

- `gpu/cuda/backend_cuda.cu`:
  - the flag, the math-mode scope and `decoder_put`;
  - templated im2col/gather kernels and `k_decoder_convtr_overlap_one`;
  - `weight_bf16` in the op, and the bf16 branches in `decoder_conv1d`, `decoder_convtr`, `decoder_conv1d_batch`
    and `decoder_convtr_batch`;
  - `decoder_ops_compatible` and the startup line.
- `tests/codec_int8_quality.py`: `--mode seanet-bf16`.
  - It compares the same CUDA binary with the flag on and off, at temperature 0, and `--batch N` compares every
    row of a gang.
  - It uses the same three measures: sample count, SNR and waveform correlation, and log-mel correlation.
  - Its bounds are provisional: 38 dB, 0.9998 and 0.998.
  - The per-pair code moved into `measure()` without changing the existing modes.
- `tools/gpu/knee.sh`: forwards `VAR=value` pairs to `ab.sh`/`serve.sh`. A knee can then run one arm with a flag;
  before this change it could not.
- `docs/cuda-serving.md`: one row in the opt-in table.

## What was verified on the Mac (no nvcc, no GPU)

- **CUDA syntax check of `backend_cuda.cu`**, host and device for sm_89: 0 errors, with the same 11 (pre-existing,
  unused-parameter) warnings as the base file.
  - Compiler: Apple clang 21 in CUDA mode (`-fsyntax-only`, Apple clang has no NVPTX backend), against CUDA 12.4
    headers taken from the Linux pip wheels (`nvidia-cuda-runtime/cublas/nvtx/cccl-cu12`) plus a
    `cudaConfigureCall` declaration shim.
  - This covers the template instantiations, the cuBLAS prototypes (`cublasGemmBatchedEx`, `cublasGetMathMode`)
    and the device code semantics. It does not cover nvcc-specific diagnostics or codegen.
- `make -k cuda cuda-server`: every C object of the CUDA tree (CLI, server, core with `-DMYNAH_ENABLE_CUDA`)
  compiles. Only the nvcc step is missing.
- `make test-c`: PASS. These are the CPU gates; the CPU tree is not changed by this branch.
- `tests/codec_int8_quality.py --mode conv --seeds 1` on `models/pocket-en`: identical numbers before and after
  the refactor (worst SNR 29.1 dB, log-mel 0.99352).
  - The existing gate FAILS on this Mac with that pack in both versions. This was found here and has nothing to do
    with this change; it should be looked at separately.
- **Not verified:**
  - nvcc build;
  - cuBLAS acceptance of bf16 A/B with fp32 C in `cublasGemmBatchedEx` (documented as supported for
    `CUBLAS_COMPUTE_32F`);
  - numerics, speed, graph capture with the flag on, VRAM.

## A/B on the L4 (g6.xlarge, sm_89)

The changes are uncommitted. Ship them as a patch onto the box checkout (`MYNAH_GPU_ROOT`, default
`/root/mynah-head`) at v1.5.0: `git -C <worktree> diff > seanet-bf16.patch`, then `git apply` on the box. Every
remote command runs with a hard timeout.

```sh
# 0. build
make cuda cuda-server CUDA_ARCH=sm_89

# 1. self-tests, flag off then on (the decoder self-test compares GPU vs CPU SEANet at 3e-3)
build/cuda/mynah-tts --gpu-self-test cuda
MYNAH_CUDA_SEANET_BF16=1 build/cuda/mynah-tts --gpu-self-test cuda
MYNAH_CUDA_KV_DTYPE=bf16 build/cuda/mynah-tts --pocket-self-check models/pocket-english-24l --device cuda
MYNAH_CUDA_SEANET_BF16=1 MYNAH_CUDA_KV_DTYPE=bf16 \
  build/cuda/mynah-tts --pocket-self-check models/pocket-english-24l --device cuda

# 2. flag off == v1.5.0: same text/seed/batch 1 from a v1.5.0 build and from this build, md5 identical
#    (in that build's tree) MYNAH_CUDA_KV_DTYPE=bf16 MYNAH_THREADS=1 build/cuda/mynah-tts \
#        --synthesize models/pocket-english-24l --text "the quick brown fox jumps over the lazy dog" \
#        --lang en --device cuda --seed 42 --output /tmp/base.wav
MYNAH_CUDA_KV_DTYPE=bf16 MYNAH_THREADS=1 build/cuda/mynah-tts --synthesize models/pocket-english-24l \
  --text "the quick brown fox jumps over the lazy dog" --lang en --device cuda --seed 42 --output /tmp/off.wav
md5sum /tmp/base.wav /tmp/off.wav      # or: python3 tools/pcm_diff.py /tmp/base.wav /tmp/off.wav

# 3. quality at temperature 0: solo path (batch 1), then the gang path (batch 8)
MYNAH_CUDA_KV_DTYPE=bf16 MYNAH_THREADS=1 python3 tests/codec_int8_quality.py \
  --binary build/cuda/mynah-tts --model models/pocket-english-24l --mode seanet-bf16
MYNAH_CUDA_KV_DTYPE=bf16 MYNAH_THREADS=1 python3 tests/codec_int8_quality.py \
  --binary build/cuda/mynah-tts --model models/pocket-english-24l --mode seanet-bf16 --batch 8
#    expect sample counts identical (SEANet is downstream of the sampler) and SNR near 43-45 dB;
#    record the numbers and set the bounds from them

# 4. 60-s knees, A then B then A again, fresh server per arm (serve.sh sets MYNAH_SERVE_PROFILE=1)
tools/gpu/knee.sh sb16-off-1 models/pocket-english-24l 128,160,192 0.88 60
tools/gpu/knee.sh sb16-on-1  models/pocket-english-24l 128,160,192 0.88 60 MYNAH_CUDA_SEANET_BF16=1
tools/gpu/knee.sh sb16-off-2 models/pocket-english-24l 128,160,192 0.88 60
tools/gpu/knee.sh sb16-on-2  models/pocket-english-24l 128,160,192 0.88 60 MYNAH_CUDA_SEANET_BF16=1
grep -E "SEANet decoder convolutions in bf16" /root/evidence/sb16-on-1-server.log      # arm check
grep -E "SERVE\]   B(128|160|192) |SERVE\] loop" /root/evidence/sb16-*-server.log       # step per width

# 5. only if the knee or the step moves: one soak per arm at the chosen level
tools/gpu/soak.sh sb16-off-soak 160 1800
tools/gpu/soak.sh sb16-on-soak  160 1800 "" MYNAH_CUDA_SEANET_BF16=1
```

**Optional:** first measure the SEANet share of the step on this engine with Nsight Systems: `MYNAH_COST_MAP=2
MYNAH_NVTX=1` under `nsys profile`, at C160, both arms. The reference engine's 49 % share does not transfer as is
(see below).

**Gate to keep the flag:**
- both self-checks PASS;
- flag-off md5 identical to v1.5.0;
- the quality gate passes at batch 1 and batch 8;
- the step at B160/B192 is lower in both ABAB pairs;
- no new failures, stalls or VRAM growth.

## Expected gain (an estimate, to be measured)

The 17 % is an upper bound here. This engine's fp32 baseline is not plain fp32: its decoder GEMMs already run
on TF32 tensor cores.

What bf16 removes:
- half the bytes of the im2col columns, which are written by one kernel and read by the GEMM. These columns are
  the largest traffic in the decoder: `C·k·L` per conv per request, e.g. 64·3·1920 floats = 1.5 MB per request
  in the last stage, ×160 requests;
- half the bytes of the convtr gather and of the weights;
- half the tensor-core time.

Unchanged: the fp32 elementwise kernels (ELU, window/tail, bias, residual, overlap and fold). All of them are
memory bound and run at fp32 width.

If the GEMM and columns traffic is ~half to two thirds of SEANet time, SEANet gets ~25-35 % faster. At a ~40-50 %
SEANet share that is a ~10-15 % step at C160-C192, so expect a knee of perhaps one level up. Next lever if the
elementwise share dominates: fuse the ELU into the im2col/gather (each ELU is a full read/write of the
activation today), and keep the residual stream in bf16 (riskier).

## Risks

1. **cuBLAS support.** If this toolkit's `cublasGemmBatchedEx` refuses 16BF/16BF→32F under `COMPUTE_32F`, the
   decoder step fails loudly (`decoder_failures`, `resident_fallbacks` in `/metrics`) instead of degrading.
   Step 1 above catches it. The fallback is a strided layout, or `cublasLtMatmul` with a batch count.
2. **Solo-vs-gang parity.** Both paths round the same inputs the same way, and only the accumulation order
   differs (GemmEx vs GemmBatchedEx, as with TF32 today). They should stay inside the 1e-3 tensor-core band of
   `--pocket-self-check`. If they do not, the self-check fails with the flag on.
3. **Quality of the narrow last stages.** The int8 work (`.work/seanet-int8-conv.md`) showed that the 32/64-channel
   stages close to the waveform reach log-mel first. bf16 (8-bit mantissa, fp32 accumulate) is far finer than
   int8, and the reference measured 43-45 dB. If log-mel fails, exclude the last stage or the last conv: a
   one-line condition in `decoder_add_op`.
4. **Small or odd shapes.** The final conv has N = 1. The leading dimensions (16/96/480/1920, out·kernel) are
   multiples of 8 and every buffer is `cudaMalloc`-aligned. If cuBLAS picks a non-tensor-core kernel for a
   shape, the result is still correct, only not faster.
5. **Graph capture.** The bf16 path allocates nothing: weights are converted at open and buffers are reused. The
   math-mode scope is host-side handle state. Graph replays are still covered only by step 4 on the box.
6. **Interactions.**
   - With `MYNAH_CUDA_FAST_MATH=1` the bf16 GEMMs still use `COMPUTE_32F` explicitly.
   - The flag is process-wide and read once, so a harness that flips the environment inside one process will not
     see the change.
   - The `bf16_weight_bytes` metric includes the SEANet copies when the flag is on.

## Not implemented: how the other two ports would go

### FlowLM bf16 GEMMs (reference: RTF p95 -8/9 %, throughput +11 %)

- **Already present:** `MYNAH_CUDA_QUANT=bf16`.
  - Mode resolution: `cuda_quant_weights_requested` in `gpu/cuda/backend_cuda.cu`, and
    `pocket_cuda_f32_qtype` / `pocket_cuda_linear_d2d` in `src/engine_pocket.c` (~1640-1700).
  - Step projections run through `mynah_cuda_matmul_bf16_d2d`: a cuBLAS BF16 GemmEx with the activations rounded
    by a separate `k_f32_to_bf16` pass.
  - Prefill and tile projections run through `tile_gemm(..., bf16)`, i.e. the SIMT `k_tile_gemm<true>`, chosen for
    batch invariance.
  - The whole mode was measured 4 % slower on the L4 (`.work/pocket-cuda-c60-l4.md`, "Final tree").
- **Why it does not reproduce the reference win (hypotheses to check):**
  - the f32 baseline here already uses TF32 tensor cores;
  - one extra activation-cast kernel and buffer per projection;
  - the bf16 tile path is SIMT;
  - the mode switches backbone, flow and Mimi together.
- **Port plan:**
  1. A per-stage switch, starting with the flow head (the reference's win), e.g. `MYNAH_CUDA_QUANT_STAGES` or
     reuse `MYNAH_QUANT_GROUPS`. The `pocket_cuda_*_qtype` helpers already split per stage.
  2. Fuse the bf16 cast into the producer of the GEMM input: the LayerNorm/RMSNorm/modulate kernels
     `k_layer_norm` and `k_flow_var_rms_norm` / `k_flow_modulate` write bf16 next to the f32 output. This removes
     the separate cast pass.
  3. Step-path GEMMs on `cublasLtMatmul` with bf16 A/B, fp32 C and a bias epilogue (drops the bias kernel).
  4. For the prefill tile, keep `k_tile_gemm<true>` for invariance, or add a bf16 WMMA variant next to
     `k_tile_gemm_tc` / `k_tile_gemm_tc2` (`MYNAH_CUDA_TILE_TC`, same fixed-order schedule).
  5. A/B with the same knee procedure. Gate: WER and speaker cosine (as in the BF16 KV quality gate,
     `.work/pocket-cuda-c60-l4.md`), because bf16 weights move the sampled trajectory.

### Shared voice-prefix KV (reference: VRAM -2 GB at 160 rows, step -10/15 %, sample-identical)

- **Today:**
  - The voice KV prefix is cached once per voice on the device: `pocket_cuda_voice_kv` in `src/engine_pocket.c`
    (~7670), stored in `state->cuda_voice_kv`.
  - It is then *copied* into each request's backbone KV by the prefill tile (`cuda_voice_pending`, ~7760-8030).
  - Every row owns `[voice positions + text + generated]` (`cuda_backbone_kv_bytes`, grown by
    `MYNAH_CUDA_KV_GROW`).
  - Batched decode attention reads per-row K/V through pointer tables: `dev_batch_k_cache`, `dev_batch_v_cache`,
    `dev_batch_positions` and `dev_batch_cache_strides` in `gpu/cuda/backend_cuda.cu`. The kernels are
    `k_self_attention[_bf16]_batch` and the fast path (`cuda_backbone_attn_fast_enabled`, ~7840-8000).
    The prefill uses `k_tile_attention[_grouped]`.
- **Port plan:**
  1. Per row, add a shared-prefix pointer and length to the batch metadata: two more device tables next to the
     ones above. The private cache then starts at position `P_voice`.
  2. Make the attention kernels read two segments: shared prefix positions `[0, P)` from the voice buffer, private
     positions `[P, pos]` from the row cache, with one online softmax across both. The order of the dot products
     stays the same, which is what keeps it sample-identical.
  3. Prefill tile: skip the voice copy, and attend to the shared prefix in the same two-segment way.
  4. Size the KV allocation, `MYNAH_CUDA_KV_GROW` and the slot pool (`MYNAH_CUDA_SLOT_POOL`) for the private part
     only.
  5. The voice buffers must be immutable and outlive every row that references them (they already live until
     engine close).
  6. Watch for any code path that rebases or rewrites positions in place (codec-window rebase is a different
     cache; check the backbone for context-window shifts).
  7. Gate: md5-identical audio at batch 1 against the copy path, VRAM at C160, then the knee.

## L4 results (2026-10-01, g6.xlarge, v1.5.0 + this diff, nvcc 12.8, sm_89)
- Build OK. `--gpu-self-test cuda`: PASS with the flag off and on.
- Flag off: `--synthesize` output md5-identical to the v1.5.0 build (ff4eb042...), so the off path is byte-identical.
- Quality at temperature 0, flag on vs off (`tests/codec_int8_quality.py --mode seanet-bf16`): solo worst SNR 47.5 dB,
  wave corr 0.999991, mel corr 0.99952; gang (batch 8) worst SNR 47.7 dB. Gate PASS.
- `--pocket-self-check` with the flag ON fails the batching check: "decode gang: request 0 of 2 differs from its own
  solo decode of frames [2, 3) at sample 861 (max_abs=0.00128)". Cause: the solo conv1d path (cublasGemmEx) and the gang
  path (batched GEMM) pick different bf16 kernels, so solo and gang round differently by ~1e-3. Fix options: route
  the solo conv1d through the same batched call (batch 1), or relax the solo-vs-gang tolerance when the flag is on.
  Must be fixed before this flag can default on.
- Closed-loop knees, 60 s, 4 voices, A/B/A/B, fresh server per arm (STREAM_RTF p95 / audio-s/s):
  | arm | C160 | C192 |
  |---|---|---|
  | off-1 | 0.858 / 156.1 | 0.879 / 152.7 |
  | on-1  | 0.839 / 159.7 | 0.849 / 155.8 |
  | off-2 | 0.872 / 153.7 | 0.883 / 150.6 |
  | on-2  | 0.836 / 162.8 | 0.850 / 155.4 |
  RTF p95 -3 % (C160) / -3.6 % (C192), speech/s +3-4 %, 0 failed. Smaller than the PyTorch engine's -17 % step, as
  expected: the conv GEMMs here were already on TF32 tensor cores; the remaining SEANet cost is the fp32 glue kernels
  (ELU / residual / fold), so the next step is fusing ELU into the im2col/gather kernels.
- Fix (same day): the solo-vs-gang check gets a BF16-codec tolerance tier (POCKET_BATCH_PARITY_*_BF16_CODEC = 3e-3,
  -50 dB; measured 1.3e-3 for conv1d alone and for convtr alone, located with the diagnostic values
  MYNAH_CUDA_SEANET_BF16=conv / convtr), and mynah_cuda_batch_invariant() reports 0 when the flag is on. L4:
  `--pocket-self-check` PASS with the flag off and on.
