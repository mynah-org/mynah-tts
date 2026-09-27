IN PROGRESS

# Pocket CUDA: realtime streaming capacity on one L4 (target C60, 24L)

## Problem

The earlier L4 ladders reported "C64/C80 clean" from memory fit and zero HTTP
errors. They are not streaming capacity:

- the server ran `--max-batch 8 --max-inflight 8`, so at C64 only eight rows
  decoded and 56 requests queued (first audio p50 12.9 s is queue wait);
- one wave per level, one fixed 3.68 s text, no warm-up record, and a tool
  that is not in the repository;
- aggregate ~6.6-7.0 audio-s/s at every level from C40 to C80 means the
  realtime knee was about 7 streams, not 40.

Offered concurrency, admitted rows, actually active decoding rows, aggregate
audio-s/s and per-request streaming RTF are five different numbers and have
to be reported separately.

## Evidence from the code audit (e640497, read-only)

Per generated frame of an N-row gang, default CUDA serving flags plus BF16 KV:

| stage | batched across rows | graph | syncs |
|---|---|---|---|
| input projection | M=N GEMM | no | 1 |
| backbone step | M=N, 24 layers | per width, recapture on new row | 1 + 12.6 MB D2H KV mirror at N=64 |
| flow/LSD | M=N | per width | 1, RNG on host |
| EOS | M=N | no | 1 |
| Mimi transformer | **no: host loop, one request, M=1** | **no** | **N** |
| SEANet decoder | batched pointer-array GEMM | keyed on exact ordered row tuple | shared |
| PCM | N D2H copies | no | 1 |

- Attention (backbone and Mimi) walks cached positions serially, one block per
  (head, row), about nine barriers per position.
- Weights are FP32 through `cublasGemmEx` with `CUBLAS_COMPUTE_32F`, so no
  tensor cores.
- Prefill runs one token per row per iteration, each token being a full
  backbone pass plus a sync, inline in the only scheduler thread.
- The server has exactly one scheduler thread that runs prefill, step, codec
  and PCM copies serially.

The opt-in multi-row Mimi path (`MYNAH_CUDA_CODEC_BATCH=1`) failed its first
waveform parity gate and has neither a graph nor BF16 KV.

## Design ideas worth testing (from public serving designs and arithmetic)

- One CUDA graph per width bucket covering a complete frame, with per-row
  lengths, masks and RNG counters in device memory, one D2H per frame.
- Fixed-slot row arena, masked lockstep stepping, per-slot state reset.
- Batched streaming conv/transformer state for Mimi and SEANet (one launch per
  layer for all rows).
- Device-resident voice prefix copied device-to-device at admission.
- Counter-based device RNG keyed by (request seed, step), preserving batch
  invariance.
- Length-aware attention kept; a split-K variant only for small widths.
- Roofline at B=64, 24L, ~300 positions, BF16 KV: about 0.6 GB weights + 1.9 GB
  KV per step, ~8-10 ms lower bound against an 80 ms frame budget. C60 is not
  a bandwidth problem; it is an overhead and serialization problem.

## Plan

1. Harness: `tools/pocket_ladder.py`, per-request JSONL, closed and Poisson
   load, warm-up discarded, /metrics deltas, nvidia-smi and server CPU/RSS.
2. Baseline ladder at real width (`--max-batch 64 --max-inflight 64`), 6L and
   24L, C1-C64.
3. Nsight Systems at C8 and at the first saturated point; per-stage cost model
   covering at least 95% of frame time.
4. Candidates in measured order; each behind a switch, parity before speed,
   ABBA on serving level.

## Acceptance gates for C60 (24L)

All 60 streams active; aggregate > 60 audio-s/s (target 75-90); streaming RTF
p95 < 1 (target < 0.9); TTFA p95 <= 500 ms (target < 200 ms); zero stalls at
250 and 500 ms buffers; zero failures, rejections and timeouts; stable VRAM;
30-minute Poisson soak with per-request records in ten windows.

## Measurements

Raw data (per-request JSONL, server logs, Nsight traces) stays on the L4 host
under `/root/evidence/`. Load: `tools/pocket_ladder.py`, closed loop, 10-15 s
warm-up discarded, 30-45 s measured, fixed text mix and seed, one client
process on the GPU host. Server: 24L, `--max-batch 64 --max-inflight 64`,
BF16 backbone KV, resident f32 weights, one scheduler thread.

### Two traps found before any measurement was valid

- `MYNAH_QUANT_GROUPS` unset on a CUDA build inherited the CPU bf16/f16 group
  defaults, which the resident kernels refuse, so backbone/flow/Mimi ran on
  the CPU oracle while the server said `device=cuda` (C1 RTF 8.7, 236 MiB).
  The CUDA default is now raw f32 and an incompatible explicit spec refuses to
  start unless `MYNAH_CUDA_ALLOW_CPU_STAGES=1`.
- `/metrics` was exactly 8192 bytes, i.e. truncated; the buffer is now 16 KiB.

### Baseline (per-request resident Mimi), 24L

| C | audio-s/s | stream RTF p50/p95 | TTFA p50/p95 ms | stalls@500 | GPU util |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.71 | 0.28 / 0.32 | 316 / 380 | 0 | 54% |
| 4 | 3.93 | 0.85 / 0.92 | 400 / 484 | 0 | 45% |
| 8 | 2.91 | 1.58 / 1.76 | 405 / 680 | 694 | 72% |
| 16 | 3.79 | 2.64 / 3.20 | 591 / 987 | 1965 | 71% |

Step cost grew ~11-12 ms per live slot at every width (B4 55 ms, B8 95 ms,
B16 171 ms): batching amortised nothing. Nsight at C8 (15 s window): GPU busy
~28%; 51% of kernel time in the per-request Mimi `k_self_attention` (15,200
calls x 140 us, one block per head walking positions serially); 7,185 stream
syncs (3.76 s blocked) and 203k kernel launches. COSTMAP at C8: Mimi
transformer 19.1 s, rest of codec 7.3 s, prefill 9.8 s, backbone 7.4 s.

### Candidate 1: cross-request Mimi tile (`MYNAH_CUDA_MIMI_TILE`, KEEP)

One call per frame for every eligible request: [requests x 16] rows through
both layers, deterministic fixed-order GEMM (a row's bits do not depend on
the batch), one warp per (row, head) causal-window attention, per-request K/V
ring in the existing device window (slot = absolute % capacity), output
scattered straight into the SEANet input. No host round trip; a request that
adopts the ring cannot fall back and fails instead.

| C | base audio-s/s | tile audio-s/s | RTF p95 base -> tile | stalls@500 | TTFA p95 ms |
|---:|---:|---:|---:|---:|---:|
| 8 | 2.91 | 10.32 | 1.76 -> 0.85 | 694 -> 0 | 680 -> 434 |
| 16 | 3.79 | 11.30 | 3.20 -> 1.20 | 1965 -> 275 | 987 -> 624 |
| 32 | - | 10.01 | - -> 2.66 | - -> 1833 | - -> 1047 |

Zero client or server failures. Step at B16 171 -> 34 ms; D2H at C16
15.4 GB -> 0.27 GB. Parity: 24L and 6L CUDA self-checks pass (solo vs gang);
CPU f32 oracle vs CUDA (same text/seed/voice, 8.16 s) is 66.4 dB SNR with f32
KV. BF16 KV alone moves the waveform to 9.1 dB SNR / 0.25 log-spectral
distance with identical length and EOS -- an open quality gate on BF16, not
on the tile. Now the default; `MYNAH_CUDA_MIMI_TILE=0` restores the old path.

Loop split after the tile at C8-C32: step 41%, prefill 38% (~100 ms slices,
token-serial, blocking every stream), other 20%. Plateau ~10-11 audio-s/s.

### Bugs fixed on the way

- Batched CUDA prefill left a row mid-tile when its neighbour finished and
  the scalar hook refused it ("non-final text flush ... not aligned"), failing
  segmented requests. The CUDA path now completes the partial tile first; the
  self-check `prefill hand-off` reproduces the exact server error without the
  fix and passes with it.
- The resident Mimi path committed its offset with an absolute check against
  a 500-position window, so every request past 31 frames silently fell back
  to the CPU oracle; the CUDA paths now use a window-relative commit.

### Retired Blackwell session (2026-09-24), kept as numbers only

The untracked `.work/blackwell-20260924/` directory held logs and an older copy
of `pocket-tts-cuda-streaming-parity.md`; no code. It was deleted on
2026-09-27 after these facts were copied here:

- the profile header records `quant=<pack default>` and `backend=<unrecorded>`:
  with the default quant spec the CUDA build kept backbone/flow/Mimi on the CPU
  oracle (the trap fixed in this campaign), so its soak numbers are not a CUDA
  baseline: 24L C64 10-min soak 5.45 audio-s/s, stream RTF p95 13.8, 100%
  stalls; 6L C64 12.3 audio-s/s;
- COSTMAP (24L, C64 wave, 65 requests): prefill 4.88 s of which 6,240
  token projections at 0.64 ms; codec 5.07 s (Mimi transformer 3.44 s =
  2.38 ms/frame, SEANet 1.59 s); backbone 1.08 s over 114 steps; flow 0.05 s;
- /health counters, same wave: 1.76 GB H2D, 303 MB D2H, 105 graph captures,
  1,467 replays, backbone width max 27 (about 18 mean).
