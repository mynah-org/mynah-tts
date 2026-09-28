IN PROGRESS (code on branch `cuda-slot-pool`, not yet measured on a GPU)

# Pocket CUDA: pool per-request resource sets instead of alloc/free per request

## Problem

On the CUDA serving path every admission (`pocket_ctx_new`) allocated, on the
single scheduler thread: pinned host staging (`step_input`, `hidden`, `denorm`,
`codec_seq`, `codec_out`, `codec_back`, `pcm` via cudaHostAlloc), the resident
backbone KV + activations, the Mimi codec KV window + activations + upsample
buffers, and a resident SEANet decoder (`mynah_cuda_decoder_open`, several
cudaMalloc per op). Every retirement (`pocket_ctx_free`) freed all of it again:
dozens of cudaFree/cudaFreeHost, plus `mynah_cuda_decoder_close` with its
stream sync. cudaFree/cudaFreeHost synchronise the whole device, so at C64
(~11 retirements/s) each retirement is a GPU bubble and scheduler time. Found
by a code audit; not yet measured by itself.

## Change

`MYNAH_CUDA_SLOT_POOL` (default on when `pocket_cuda_resident_requested`, i.e.
backend `cuda` and `MYNAH_CUDA_RESIDENT` not 0; `=0` restores the old path
exactly). The CPU path never reaches the pool (the gate is the backend name).

- `pocket_ctx_free`: after the existing drain, `pocket_cuda_slot_park` moves
  the ctx's device buffers, decoder handle and pinned staging into a
  `pocket_cuda_slot` on a mutex-protected free list in `mynah_engine_state`
  (cap `POCKET_MAX_BATCH` idle sets; beyond it the set is freed as before).
  The decoder's graphs are forgotten at park, as close did; `mynah_cuda_graph_forget`
  now also drops the cross-request decoder batch graphs naming that decoder
  (otherwise the 128-entry batch-graph cache would fill with retired gangs).
- `pocket_ctx_new`: takes the idle set whose backbone KV fits best (bytes >=
  need, same KV dtype), else the largest one (only its KV is re-allocated).
  The alloc helpers consume the parts they need; leftovers are parked again.
  Failure paths go through `ctx_free`, so a taken set always returns to the
  pool or is freed.
- Reset on take: pinned host buffers memset to 0 (the old path was calloc-clean
  or fresh pinned pages); Mimi KV window, upsample tail, decoder input/output
  zeroed on the stream; `mynah_backend_decoder_reset` on the decoder (the
  prologue still resets decoder + upsample tail as before). Backbone KV is not
  cleared by default (attention only reads positions the request wrote; a
  fresh cudaMalloc was never cleared either); `MYNAH_CUDA_SLOT_POOL_ZERO_KV=1`
  clears it for the leak A/B.
- The backbone KV layout still uses `cuda_backbone_capacity` (= this request's
  `max_seq_len`) as stride; a larger reused buffer just has an unused tail.
  `cuda_backbone_kv_bytes` records the real size.

Not changed: the host backbone KV `calloc` in `transformer_ar_state_new` for
device-owned rows. Device ownership is decided later, at the prologue, and can
be revoked (segments, fallback), and the host state is also the CPU fallback,
so skipping it needs a lazy-allocation mode in `transformer_ar`; a large calloc
is mmap-backed and untouched pages cost no RSS, only the mmap/munmap.

Not changed: the drain (`cudaStreamSynchronize`) before park. The backend has
one stream, so device reuse is stream-ordered already; the drain protects the
pinned host buffers from a still-queued async copy. Replacing it with an event
needs a backend event API.

## Acceptance gate (L4, same flags, ABAB)

1. Leak check: same seed/text/voice, a request admitted right after another
   retires, with `MYNAH_CUDA_SLOT_POOL=0`, `=1`, and `=1
   MYNAH_CUDA_SLOT_POOL_ZERO_KV=1`: PCM must be bit-identical across the three.
2. `--pocket-self-check MODEL --device cuda` PASS with the pool on.
3. C64 ladder + soak: stalls, RTF, TTFA, scheduler CPU% vs `MYNAH_CUDA_SLOT_POOL=0`;
   nsys should show no cudaFree/cudaFreeHost/decoder close on retirement.
   KEEP only if not worse on any gate.
4. Idle device memory after a burst stays at the peak working set (pool keeps
   up to 128 sets until the engine state is freed); check nvidia-smi.
