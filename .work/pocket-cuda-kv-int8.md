DONE 2026-10-04 (code default; 30-minute soak qualification and WER pending)

# int8 backbone KV for the Pocket CUDA engine (`MYNAH_CUDA_KV_DTYPE`)

## Problem

At C256-C288 on one L4 (24L, 2026-10-02 defaults) the batched decode attention
reads every row's KV once per layer per step, and the BF16 rows are most of the
device memory (peak 21.3 GB at `--max-batch 288`). Halving the bytes per
position attacks both: attention bandwidth and the VRAM ceiling.
`.work/pocket-cuda-kv-tiering.md` already named int8 KV as the cheaper lever.

## Layout

- Each stored position of a K or V plane is one **record**: `heads * head_dim`
  int8 values followed by `heads` float scales. Value `d` of head `h` is
  `q[h * head_dim + d] * scale[h]`, `scale = max|x| / 127` over that head and
  position (symmetric, round to nearest even, clamped to [-127, 127]).
- 24L (16 heads x 64): 1024 + 64 = **1088 bytes** per position and plane,
  against 2048 for BF16 (-47 %). Records are a multiple of 16 bytes for every
  Pocket model (`heads * head_dim % 16 == 0`, `heads % 4 == 0`), so a lane
  reads 8 values plus the scale with one 16-byte load.
- Rows stay plane-major (`[layer][K|V][position][record]`); `kv_strides` pitch
  and V offset are in bytes. The shared voice prefix stays BF16 and is read
  through the shared-voice path, so int8 only covers what each request writes.
- Writers: the prefill tile (`k_tile_rope_store_i8`: RoPE, then per-head
  quantisation, one warp per head) and the batched split decode attention
  (`k_self_attention_i8_batch_split`: the new K/V is quantised before the
  block reads it, so the current position is read exactly as later steps will
  read it). Same partition, online softmax and merge order as the BF16 split
  kernel: deterministic and batch-invariant.
- The single-row step runs the batched kernel with one row; padding rows get
  2 records per layer inside the existing pad buffer.

## Constraints (resolved once at model load)

- Default when `MYNAH_CUDA_KV_DTYPE` is unset (or `int8`), and only when the
  resident backbone, `MYNAH_CUDA_PREFILL_TILE`, `MYNAH_CUDA_SHARED_VOICE` and
  `MYNAH_CUDA_SHARED_VOICE_STRIP` are on and heads are 64 wide (at most 32, a
  multiple of 4). Otherwise the rows stay BF16 (a warning when int8 was asked
  for explicitly). `=bf16` is the rollback, `=f32` the CPU/GPU parity oracle.
- Records have no host mirror: int8 rows are device-owned, no host shadow and
  no CPU retry; the host-cache upload is refused.
- A row without a shared model-voice KV (`state->voices[speaker].kv == NULL`,
  e.g. a file-backed voice: `kv_skip == 0`) has no voice prefix to skip, so it
  cannot use the records: its backbone runs on the CPU (logged once). Voices
  in the pack, built-in or written by `mynah_engine_pocket_clone_voice`, are
  model voices and use int8. `MYNAH_CUDA_KV_DTYPE=bf16` is the rollback for
  workloads with such rows.
- `MYNAH_CUDA_KV_VMM` (position-major) is ignored with int8.

## Measurements

One NVIDIA L4 (24 GB, sm_89) on Vast.ai, 24L English pack, built from this
tree, `--gpu-self-test cuda` PASS (it has no int8-KV case yet). 2-minute closed-loop knees
(`tools/pocket_ladder.py`), v2 corpus, voices alba, marius, javert, jean, seed
1234, `--max-batch` = top level, `MYNAH_QUANT_GROUPS=none MYNAH_THREADS=1`,
8 HTTP workers, every other default on. KV dtype is the only change.

| KV | C | stream RTF p95 | audio-s/s | TTFA p95 | stalls 250/500 | failed |
|---|---:|---:|---:|---:|---|---:|
| bf16 | 256 | 0.745 | 314.6 | 129 ms | 0 / 0 | 0 |
| bf16 | 288 | 0.827 | 315.9 | 143 ms | 0 / 0 | 0 |
| int8 | 256 | **0.681** | **345.4** | 118 ms | 0 / 0 | 0 |
| int8 | 288 | **0.782** | 334.8 | 134 ms | 0 / 0 | 0 |
| int8 | 320 | **0.855** | 338.4 | 146 ms | 0 / 0 | 0 |

Peak VRAM: bf16 21.3 GB at `--max-batch 288`; int8 16.3 GB at
`--max-batch 320`, server ready in 52 s. Audio seconds per request are equal
between the two modes (no EOS drift).

Quality: no WER run on this change (intentionally skipped for the screen);
equal audio seconds per request is the only quality signal measured. The
30-minute soak qualification with captured audio and WER
(`tools/gpu/qualify.sh`) is the gate still owed.

Not run: `--pocket-self-check` with the int8 default (the documented
self-check keeps `MYNAH_CUDA_KV_DTYPE=bf16`); 6L throughput with int8 (the 6L
profile pins bf16).

## Decision

int8 becomes the code default for the CUDA Pocket backbone KV; the L4 24L
profile runs it at `--max-batch 320` (screening ceiling). C160 stays the
soak-qualified point until the int8 soaks land.
