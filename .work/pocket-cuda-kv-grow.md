IN PROGRESS (code on branch `cuda-kv-grow`, not yet measured on a GPU)

# Pocket CUDA: grow the device backbone KV instead of sizing it for the worst case

## Problem

Per-stream VRAM on the CUDA serving path is ~178 MB, ~165 MB of it the BF16
backbone KV. `pocket_ctx_new` sizes it at voice + text + (max_steps + 1)
positions, and `max_steps` defaults to 1500 frames (120 s) because model.json
declares no `max_decoder_steps`. 24L BF16 is 96 KiB per position, so ~1677
positions per request while a typical request touches ~300. C128 runs out of
memory on a 23 GB L4 because of this worst case, not because of live data.

## Change

`MYNAH_CUDA_KV_GROW` (default on; `=0` restores the full-capacity allocation
exactly). Only the device cache changes; the host transformer state (the CPU
path) keeps its full `max_seq_len` and is untouched.

- Initial device capacity, for a context expected to take the prefill-tile
  (device-owned) path: voice + text ceiling + min(max_steps + 1,
  round_up(max(256, 3 * text + 64), 256)). Pocket emits 1.96..2.57 frames per
  token (table in `pocket_ctx_new`), so an ordinary request never grows. Other
  contexts keep the full capacity.
- `pocket_cuda_backbone_reserve(ctx, needed)`: grows in 256-position chunks
  (capped at the full capacity): allocate, copy each layer's live K and V
  prefix (host offset positions) into the wider [layer][K/V][capacity][dim]
  layout, sync, free the old cache. All-or-nothing; runs outside any capture.
- Writers guarded: the step pre-flight in `pocket_step_batch` (the normal
  trigger), `pocket_cuda_backbone_step_batch` (before `batch_begin`), the
  single-row `pocket_cuda_backbone_step`, `pocket_cuda_prefill_tile` (text and
  voice copy), and `pocket_cuda_backbone_upload`. `reserve_text` already
  re-allocates through `pocket_cuda_backbone_alloc`.
- Nothing bakes the old pointer or capacity: the batched step rebuilds its
  per-layer K/V pointer tables every step (graph replay reads those host
  tables, strides are `attn_dim`), the prefill tile rebuilds kv/rings per call.
- Slot pool: a parked cache is reused only when it holds 1x..2x the need;
  otherwise the largest set is taken and its cache re-allocated (the oversized
  one is freed). A reused larger cache is laid out over all its bytes.
- Failure: a device-owned row that cannot grow fails the step; the driver's
  `step_isolate` retires only that row. A host-shadowed row falls back to CPU.
- Not done: trimming the pool to the in-flight high-water mark.

## Acceptance gate (GPU, not yet run)

1. `./build/cuda/mynah-tts --pocket-self-check models/pocket-english-24l --device cuda`.
2. A long utterance that grows several times (`MYNAH_CUDA_KV_GROW_LOG=1`
   prints each growth) is bit-identical to `MYNAH_CUDA_KV_GROW=0`, same seed.
3. VRAM at C64/C96/C128 on the L4; C128 must no longer OOM.
