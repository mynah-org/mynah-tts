# Pocket CUDA: first-burst latency on a fresh server (cold slot pool, per-width graph capture)

Status: in progress, 2026-09-30. Branch `pocket-cuda-cold-burst` (from v1.5.0).

## Problem
On a freshly started CUDA server, the first burst of simultaneous streaming requests gets its first audio late,
and a large share of that delay is cold state rather than steady-state cost. The server is v1.5.0 with the L4 24L
profile (`--max-batch 160 --max-inflight 160`, bf16 KV) on one L4 (g6.xlarge, 4 vCPU). The measurement is a wave of
160 requests at t=0, 2 waves (`tools/serving_profile.py --mode wave`):

| State | TTFB p95 | TTFA p95 | backbone+flow graph captures during the waves | decoder graph captures |
|---|---|---|---|---|
| Fresh server | 737 ms | 1,179 ms | 2 → 265 | 0 → 136 |
| After one discarded wave | 210 ms | 647 ms | +98 | +52 |
| Fresh, `--warmup 160` | 738 ms | 1,195 ms | — | 0 |

In steady state (closed loop C160) TTFA p95 is ~148 ms, so the burst penalty is admission-time work.

## Evidence (code, v1.5.0)
- **Serial admission allocations.** Admission (`src/inference.c` ~1245-1294) calls `slot_start` → `pocket_ctx_new`
  once per request, serially on the scheduler thread. On a fresh server the slot pool (`MYNAH_CUDA_SLOT_POOL`,
  `src/engine_pocket.c` ~4370-4525) is empty, so every admission allocates device state. This is the TTFB spread:
  737 ms cold against 210 ms warm.
- **Graphs keyed by exact batch width.** The backbone, prefill, flow and decoder graphs are all keyed by exact width
  (`src/engine_pocket.c` ~6595, 6883; `gpu/cuda/backend_cuda.cu` ~5685-5720). A burst walks many widths (1…160), and
  each new width is captured during traffic. The start-up warm-up is one request, so only width 1 is warm.
- **`--warmup N`** sends requests through the queue one at a time. It warms neither widths nor the pool.

## Plan (each part behind its own flag, CUDA path only, default off until measured)
1. **Slot-pool prefill at start-up.** Build `N` slot sets in the pool before the server accepts traffic.
   `N` defaults to `--max-inflight`; the flag is `MYNAH_CUDA_SLOT_POOL_PREFILL=0|1|N`. First-burst admission then
   takes parked sets instead of allocating.
   - Gate: fresh-server wave TTFB p95 close to the warm value (~210 ms).
   - Report the start-up time and VRAM cost.
   - Outputs md5-identical with the flag off and on (the pool already reuses sets).
2. **Width-bucketed graphs.** Pad the step batch to the next bucket (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160
   … up to `--max-batch`) with inert rows, and capture every bucket at start-up.
   - Gate: zero graph captures during traffic.
   - Fresh-server wave TTFA p95 ≤ the warm value; steady-state throughput within noise at C160.
   - Outputs identical per row to the unpadded path.

## Acceptance
- Fresh-server wave (160 at t=0, 2 waves): TTFA p95 < 500 ms with parts 1 and 2 on, or a written explanation of the
  remainder.
- C160 closed-loop screen unchanged within noise.
- `make cuda` clean build, `--pocket-self-check` PASS, and the unit tests the repo runs for the touched paths.

## Related, not changed here
`MYNAH_CUDA_PREFILL_FIXED=0` (cuBLAS/TF32 tile GEMM) cuts warm burst TTFA p95 from 647 to ≤500 ms. The fixed-order
path exists for batch invariance, so switching it is a product decision; it is recorded here only as a measurement.

## Part 1 result (2026-09-30, L4 g6.xlarge, fresh server per arm, same profile command)
Implementation: `server/main.c` (+99 lines, no engine change). `MYNAH_CUDA_SLOT_POOL_PREFILL` = unset/0 off; 1 = one set
per `--max-inflight` slot; N = N sets (clamped to `--max-inflight`); ignored unless `--device cuda`. After the
ordinary warm-up and before the accept loop, N synthetic requests of one ~first-segment sentence are queued at once
(marked warm-up, so they stay out of `jobs.completed`), and on retirement each parks its set in the existing slot pool.

| Arm | Start-up | VRAM after start | Graph captures after start (backbone/decoder) | Wave 160×2: TTFB p95 | TTFA p50 / p95 | Graph captures during waves + ttsbench | ttsbench c160 (after the wave) |
|---|---|---|---|---|---|---|---|
| off | 1.5 s | 2.1 GB | 2 / 0 | 738 ms | — / 1,179 ms (header lead 515/1,090) | +263 / +136 | 433 / 725 ms |
| on (160) | 16.9 s (+15.4 s) | 10.2 GB | 164 / 0 | 599 ms | — / 997 ms (header lead 583/954) | +260 / +134 | 432 / 723 ms |
| reference: warm (one discarded wave first) | — | — | 174 / 85 | 210 ms | — / 647 ms | +98 / +52 | 399 / 661 ms |

- Equivalence: with the flag off and on, 3 texts at batch 1 (seed 42, alba) give identical PCM md5s. `--pocket-self-check`
  (bf16 KV) passes on the clean build.
- Reading: a prefilled pool removes part of the first-burst admission cost (TTFB p95 738 → 599 ms, TTFA p95
  1,179 → 997 ms). It is far from the warm state (210 / 647 ms), however. The same ~260 backbone/flow and ~135
  decoder graph captures still happen during the first waves: parking forgets the decoder graphs (by design), and the
  prefill passes only the widths its own batch walks through. The main cold cost is per-width graph capture, which is
  part 2 (width-bucketed graphs captured at start-up). Part 1 alone is not worth +15 s start-up and should stay off
  by default. Keep it as a building block for part 2.
- Not run: `make test` / `tests/test_server.sh` (they need the CPU build and more time); the `warmup` byte-identity
  gate of test_server.sh was not re-run.

## Next: first-chunk latency in steady state (goal: match or beat a single-graph continuous-batching server)
At C160 on the same L4 class, a continuous-batching PyTorch server that captures one CUDA graph per step
(backbone step + flow head + codec transformer + SEANet + PCM16, one host sync per frame) measures TTFA p50/p95
81/136 ms in a 30-min soak at 199 audio-s/s. This engine measures 137/154 ms (qualA, 30 min, 184.5 audio-s/s) and
~148 ms p95 on the same g6.xlarge (2-min screen). The p95 gap is small. The p50 gap is ~56 ms, about one B160 step
(~64 ms). All three items are hypotheses to measure first:

1. **One extra pipeline step before the first chunk.** Here the decoder gang/lane runs as its own pass. If frame N
   is decoded and emitted in iteration N+1, the first chunk costs one extra step.
   - Measure: per request, the number of scheduler iterations from admission to the first emitted chunk (add it to
     the `[SERVE]` profile, or derive it from existing timestamps).
   - Fix if confirmed: decode and emit the first frame of newly admitted rows in the same iteration.
2. **Prefill tile GEMM on the TTFA path.** `MYNAH_CUDA_PREFILL_FIXED=1` (the default) runs the fixed-order FP32 SIMT
   tile GEMM with no tensor cores. With `=0` (cuBLAS/TF32), warm-burst TTFA p95 drops from 647 to ≤500 ms.
   - Options: a fixed-order tensor-core tile (keep batch invariance, and remove today's `MYNAH_CUDA_TILE_TC`
     slowdown), or a serving profile that opts out of batch invariance.
3. **Host syncs per step.** `/health` shows ~7 sync calls per backbone batch step (5,628 syncs over 811 steps), where
   the reference server makes 1 per frame.
   - Measure the host gap per step (`[SERVE]` other/idle share).
   - Fix if significant: fold the syncs into one per frame (events instead of stream syncs), as in the E15 notes.

Acceptance for this block: at C160 in steady state, TTFA p50 ≤ ~90 ms and p95 ≤ 140 ms with throughput within noise.
The fresh-server first burst must already be fixed by parts 1–2.

## Part 2 result (2026-09-30, L4 g6.xlarge, fresh server per arm, same profile command)
Implementation, `MYNAH_CUDA_WIDTH_BUCKETS` (unset/0 off; 1 = 1,2,4,8,16,24,32,48,64,96,128,160,192,256,384; or an
ascending comma list):
- **Seam:** one helper, `pocket_cuda_exec_width(count, capacity)` in `src/engine_pocket.c`. The cross-request
  backbone step (plain, condition-input and prefill graph keys) and the flow step run at the smallest bucket ≥ the live
  width, clamped to the scratch capacity. The graph key, device work and staging layout use that width.
- **Pad rows:** they sit at position 0 of a private one-position KV (`scratch->cuda_pad_kv`, allocated on first use)
  and take a copy of row 0's input. Nothing is read back from them, and host commits cover only live rows.
- **Not bucketed:** the Mimi decoder gang. Each of its rows owns causal rings, so there is no inert row to pad with;
  it keeps exact widths plus REUSE.
- **Server** (`server/main.c`): after the warm-up, `--max-inflight` synthetic requests start together, and request i
  stops after 4 + 2i steps. The live width therefore walks from full down to 1, and every bucket is captured with both
  input-staging variants. One log line reports graphs, ms and memory. The walk reuses the part-1 runner.

| Arm | Start-up | VRAM after start | Graphs after start (bb+flow / decoder) | Wave 160×2: TTFB p95 | TTFA p95 (header lead p50/p95) | Graphs added by the waves + ttsbench | ttsbench c160 p50/p95 | 2-min C160 screen |
|---|---|---|---|---|---|---|---|---|
| off | 1.4 s | 2.1 GB | 2 / 0 | 752 ms | 1,195 ms (527/1,091) | +264 / +137 | 438 / 736 | 170.7 audio-s/s, RTF p95 0.845, TTFA p95 148, 0 stalls |
| buckets | 45.8 s | 13.4 GB | 195 / 0 | 630 ms | 1,051 ms (618/1,000) | **+6** / +135 | 415 / 685 | 167.7 audio-s/s, RTF p95 0.861, TTFA p95 153, 0 stalls |
| buckets + pool prefill | 63.4 s | 13.7 GB | 355 / 0 | **190 ms** | **691 ms** (521/653) | +6 / +135 | 416 / 680 | — |
| reference: warm (part 1) | — | — | — | 210 ms | 647 ms | +98 / +52 | 399 / 661 | — |

- **Equivalence:** batch 1 (3 texts, seed 42) is IDENTICAL with the flag off and on. Batched runs (3 and 5
  concurrent) are not bit-identical: live rows take a different path through the width-dependent GEMM tiling at the
  padded width, and sampled AR trajectories then diverge. The engine is not batch-invariant without the flag either:
  two flag-off runs at 3 concurrent already differ (SNR 7–53 dB). The pad rows write only their private KV.
- **Self-check:** `--pocket-self-check` (bf16 KV) PASSES with the flag off and on.
- **Reading:**
  - Buckets remove the backbone/flow capture cost during traffic (+264 → +6), but alone they gain little on a fresh
    burst (TTFA p95 1,195 → 1,051). Together with the pool prefill, the fresh-server burst reaches the warm state
    (TTFB p95 190 vs 210, TTFA p95 691 vs 647): the cold-start penalty is gone.
  - What is left is warm cost: the fixed-order FP32 prefill tile plus the full-width first step
    (`MYNAH_CUDA_PREFILL_FIXED=0` measured 432 ms warm), and ~135 decoder-gang captures.
  - The padding cost is ~2 % throughput at C160 in this short screen (167.7 vs 170.7, RTF p95 0.861 vs 0.845).
  - The start-up walk costs ~44 s, and the VRAM after start is mostly the parked slot sets, not graphs.
- **Recommendation:** keep both flags off by default. For a serving profile where cold-start bursts matter, use buckets
  + pool prefill, and accept ~1 min of start-up.
- **Next:**
  - shorten the walk (stagger 1, or one step per bucket);
  - bucket or pre-record the decoder gang (per-bucket pre-recorded gangs with REUSE);
  - decide the prefill GEMM question.
- **Not run:** `make test` / `tests/test_server.sh` (they need the CPU build).

### Measured 2026-09-30 (v1.5.0 baseline build, warm, closed loop, server-side means from `/metrics`)
| C | Queue wait (created → admitted) | Admitted → first audio | Server TTFA mean | Step (≈ RTF × 80 ms) | TTFA in steps |
|---|---|---|---|---|---|
| 160 | 60.7 ms | 107.5 ms | 168.2 ms | ~66–69 ms | ~2.5 (0.9 + 1.6) |
| 64 | 26.9 ms | 43.1 ms | 70.0 ms | ~33 ms | ~2.1 (0.8 + 1.3) |

The single-graph reference server takes ~1.2–1.3 steps in total at C160 (soak TTFA p50 81 ms). The gap of about one
step splits roughly in half:
- **Slot release to admission** (~0.9 step at `--max-inflight == C`): a retired slot is freed at the next frame
  boundary, and admission runs once per iteration.
- **Admission to first chunk** (~1.6 steps instead of ~1).

`[SERVE]` did not print on SIGINT, even after waiting 90 s. The next step is per-request iteration counters behind
`MYNAH_SERVE_PROFILE` (created → admitted → first chunk emitted, plus the iteration phase at admission), to separate
prefill, step and decoder lag. Candidate fixes after that:
- free and admit within the same iteration as a retirement;
- emit the first frame in the admission iteration.

## Part 3 — steady-state first chunk (2026-09-30, L4 g6.xlarge, fresh server per arm, v1.5.0 L4 24L profile)

### Instrumentation (behind `MYNAH_SERVE_PROFILE`; nothing runs or allocates when it is unset)
- `graph.h`: optional `sink.phase(ud, iteration, phase)`. The driver reports each iteration's boundaries:
  admission, prefill, step+emit+decode, and retire.
- `server/main.c`: per request, it records the arrival offset inside the current iteration, whether a slot was free
  at arrival, and the iteration index at admission and at the first chunk.
- `GET /debug/first-chunk`: cumulative 5-ms histograms (difference two reads), readable while the server runs.
- Why `[SERVE]` never printed on SIGINT: the summary prints only after the scheduler returns, and shutdown joins the
  HTTP workers first. Not fixed here. The likely cause (not verified) is workers blocked on idle keep-alive client
  connections. The debug endpoint removes the need for a shutdown.

### What the instrumentation found (flag off, warm, 60-s closed loop; p50 of 5-ms buckets)
| | C160 | C64 |
|---|---|---|
| Iteration (≈ step), p50 | 62.5 ms | 22.5 ms |
| Arrival offset after the iteration start, p50 | 7.5 ms | 2.5 ms |
| Arrivals that found a slot free | 100 % | 100 % |
| Queue wait, p50 | 57.5 ms | 32.5 ms |
| Admitted → first chunk, p50 | 67.5 ms | 37.5 ms |
| First chunk in the admission iteration | **100 %** | **100 %** |
| TTFA (server), p50 / mean | 127.5 / 163 ms | 67.5 / 71 ms |

- **Hypothesis 1 (an extra pipeline step before the first chunk) is false.** Every first chunk is emitted in the
  iteration that admitted the request.
- The extra ~1 step is **queue wait with a free slot**. A closed-loop client sends its next request as the previous
  one completes. Completion happens at retirement, at the end of an iteration. The new request reaches the queue
  ~2–8 ms later, just after the next iteration's admission point, and waits one full step: TTFA ≈ (T − offset) + T.
- This is phase locking to the retirement point. Open-loop (Poisson) arrivals would wait T/2 on average.
- The p95 tails in these windows (admitted → first chunk 317 ms) come from the ramp of the measured run. It starts
  after the first snapshot, so it is a burst, not steady state.

### Fix: `MYNAH_CUDA_FAST_FIRST_CHUNK` (CUDA serving only, opt-in; unset = unchanged)
- The admission loop was lifted unchanged into `admit_pass()` (`src/inference.c`).
- New optional `sink.wait_arrival` (`graph.h`). The server installs it only with `--device cuda` and the flag set.
- When a slot is free, the driver runs a second admission and prefill pass right before the step. New arrivals join
  this step instead of the next.
- `MYNAH_CUDA_FAST_FIRST_CHUNK_WAIT_US` (default 0, max 20000) holds the step for up to that long for an arrival.
  It applies only when a slot retired in the previous iteration.
- Batch-1 PCM md5: identical, flag off vs on (3 texts, seed 42). Self-check: PASS.
- Batched runs are not batch-invariant even with the flag off, so they are not compared by md5.

| Arm | C | Client TTFA p50 | TTFA p95 (steady, soak report) | Server queue wait mean | Server TTFA mean | STREAM_RTF p95 | audio-s/s |
|---|---|---|---|---|---|---|---|
| off | 160 | 134 | 149 | 58.5 | 163.0 | 0.840 | 157.7 |
| on | 160 | 138 | 157 | 56.7 | 166.5 | 0.867 | 151.8 |
| on, wait 3 ms | 160 | 133 | 157 | 35.7 | 151.7 | 0.892 | 146.7 |
| on, wait 8 ms | 160 | **77** | 152 | 11.2 | 121.8 | **0.932** | 143.0 |
| off | 64 | 69 | 79 | 27.1 | 71.0 | 0.411 | 142.4 |
| on | 64 | 69 | 84 | 22.8 | 69.9 | 0.413 | 141.7 |
| on, wait 3 ms | 64 | **39** | 61 | 2.4 | 47.2 | 0.447 | 135.8 |
| on, wait 8 ms | 64 | **39** | 49 | 1.8 | 46.8 | 0.447 | 135.8 |

Bursts, C160 servers:
- Wave 160×2: TTFA p95 636 / 638 / 634 ms (off / on / on + 3 ms). Unchanged.
- External streaming bench (`ttsbench`-style closed loop), `--procs 4`, c160 (aggregate · opening burst · steady p50/p95):
  | Arm | Aggregate | Opening burst | Steady | Underrun requests |
  |---|---|---|---|---|
  | off | 450/1194 | 465/1196 | 137/179 | 19 |
  | on | 385/1350 | 400/1353 | 133/175 | 11 |
  | on, wait 3 ms | 606/621 | 615/623 | 142/166 | 5 |

### Verdict
- The late pass alone does nothing measurable. Arrivals land after it, because the admission-to-step window is only
  ~3 ms.
- The hold works, but it costs step time one-for-one:
  - **C64 (lots of RTF headroom): a clear win.** With a 3 ms hold, TTFA p50 goes 69 → 39 ms (−43 %) and p95 79 → 61
    ms, for −4.6 % throughput at RTF p95 0.45.
  - **C160: not acceptable.** An 8 ms hold halves p50 (134 → 77 ms, better than the reference 81), but RTF p95
    0.840 → 0.932 fails the 0.90 gate, at −9 % throughput. A 3 ms hold gives nothing at C160 (arrivals come at ~7.5
    ms) and still costs RTF.
- Recommendation: keep the flag off by default. Use `FAST_FIRST_CHUNK=1 WAIT_US=3000` for deployments that run well
  below their knee (≤ ~C96 on an L4).
- The steady-state gap to the single-graph reference server at C160 is mostly this closed-loop phase lock. In
  production, arrivals are not locked to retirements, so the expected wait is about T/2.
- A cost-free fix needs requests to be able to join a step already in flight. That means an admission and prefill
  path that overlaps the running step, not a hold. Out of scope here.

## Part 4 — admission cap, fixed-order TF32 tile, admission cost (2026-09-30)

**Changes, all behind flags that default to off:**
- `MYNAH_ADMIT_PER_ITER=N` (`src/inference.c`): at most N new requests per scheduler iteration. The cap covers the
  top-of-iteration and the late admission passes together.
- `MYNAH_CUDA_TILE_TC=2` (`gpu/cuda/backend_cuda.cu`, `k_tile_gemm_tc2`): the fixed-order TF32 prefill tile at
  128x128 per block, 8 warps, with the next K slab prefetched into registers. The accumulation order is exactly
  `k_tile_gemm_tc`'s (same split-K, 32-wide slabs, 8-wide MMAs, TF32 rounding, 16-aligned fragments). `=1` keeps
  the old 64x64 kernel.
- `[CTX]` lines under `MYNAH_SERVE_PROFILE` (`src/engine_pocket.c`): per-section cost of `pocket_ctx_new`, printed
  every 160 contexts.

**Checks:**
- Clean `make cuda cuda-server` for sm_89.
- `--pocket-self-check` with bf16 KV passes for the default, `TILE_TC=1` and `TILE_TC=2`.
- md5 at batch 1: identical with the cap off and at 32; `TILE_TC=1` and `=2` identical over 4 texts, seed 42.

**Wave, 160 at t=0, 2 waves:** TTFB and TTFA p50/p95 in ms, STREAM_RTF p95, and whether the server was fresh or
had already served one wave (warm).

| Arm | Fresh | Warm |
|---|---|---|
| default (fixed-order SIMT tile) | TTFB 155/744, TTFA 1181/1202, RTF 0.657 | TTFB 104/210, TTFA 602/658, RTF 0.645 |
| `TILE_TC=1` | TTFA 1102/1117 | TTFA 532/589 |
| `TILE_TC=2` | TTFA 1115/1132 | TTFA 544/602 |
| `PREFILL_FIXED=0` (cuBLAS reference) | TTFA 960/979 | TTFA 401/438 |
| cap 16 | TTFB 457/1158, TTFA 539/1249, RTF 0.933 | TTFA 368/810, RTF 0.833 |
| cap 32 | TTFA 511/1232, RTF 0.904 | TTFA 381/715, RTF 0.862 |
| cap 64 | TTFA 483/1210, RTF 0.840 | TTFA 483/688, RTF 0.835 |
| pool prefill + width buckets + `TILE_TC=2` | **TTFB 97/203, TTFA 496/593** | TTFA 514/535 |
| pool prefill + width buckets + cuBLAS | TTFA 399/416 | TTFA 360/372 |

- ttsbench `--procs 4` at c160 on the full `TILE_TC=2` stack: aggregate 529/544, burst 538/547, steady 147/201,
  22 underrunning requests.
- Steady guard, 60 s closed loop at C160:
  - default: 152.7 audio-s/s, RTF p95 0.861, TTFA p95 152 ms;
  - full `TILE_TC=2` stack: 138.5 audio-s/s, **RTF p95 0.916** (fails the 0.90 gate), TTFA p95 157 ms.

**Reading:**
1. **The admission cap trades p95 for p50 and hurts pacing.** The first rows get audio sooner (header lead 74–240 ms),
   but the tail gets worse. Every later admission iteration adds a prefill tile to steps that already stream, so
   STREAM_RTF p95 rises to 0.83–0.93. Keep it off.
2. **The prefill GEMM kernel is not the burst bottleneck.** `TILE_TC=2` is bit-identical to `=1` and no faster end to
   end (the tile is a small share). cuBLAS (`PREFILL_FIXED=0`) is ~200 ms faster warm, and the TF32 tiles recover
   only ~60 ms of that. The cuBLAS path also drops the fixed-order tile schedule, so the gap is prefill structure,
   not GEMM throughput. Open item: time the prefill tile pass with events per layer, both modes.
3. **Admission is serial host work on the scheduler thread:**
   - The server phase totals give ~1,120 ms cold and ~590 ms warm of admission per 160-request wave, i.e.
     ~7 / ~3.7 ms per request.
   - Warm-pool `[CTX]` breakdown, mean per context, of 1.37–2.2 ms in total: `ar_states` 0.45–0.97 ms (host AR state
     buffers), `codec_setup` 0.53–0.87 ms (SEANet/codec host state), CUDA backbone/codec/decoder 0.13–0.35 ms,
     voice/host buffers ≤0.1 ms.
   - The remaining ~1.5–2 ms per request is the sink's `next_job`: starting the stream writer (one thread per stream)
     and sending the header.
   - This is what TTFB p95 measures, and the cap cannot reach it.
4. **Combining the flags fixes the cold start:** pool prefill + buckets + TF32 tile take a fresh server to TTFB
   p95 203 ms and TTFA p95 593 ms, from 744 / 1,202. But the steady guard regressed (RTF p95 0.916 vs 0.861 on single
   60-s runs), so it needs a longer A/B before any default changes. Part 2 measured the bucket padding at ~2%.

**Next (not done):**
- **Asynchronous admission** (`MYNAH_ASYNC_ADMIT`): build contexts ahead of the scheduler, i.e. host AR/codec
  states, stream writer and header, on a helper thread or in the HTTP worker as soon as a job is queued. The
  scheduler then only takes ready contexts. Expected to remove ~3.7 ms × N of serial time from every burst (~590 ms
  at N=160) and most of the TTFB spread.
- Profile the prefill pass itself (fixed vs cuBLAS) to find the ~200 ms structural gap.

## Part 6 — asynchronous admission, `MYNAH_ASYNC_ADMIT` (2026-09-30)

**Change** (flags off by default; with them unset the code path is the old one):
- `src/tts_engine.h`: two optional hooks APPENDED to the vtable. `ctx_new_host` does the host-only half of context
  creation and may run on any thread. `ctx_attach` does the slot-pool take, the pinned buffers and the device
  allocations, on the scheduler thread.
- `src/engine_pocket.c`: `pocket_ctx_new` split into `pocket_ctx_create(..., host_only)` +
  `pocket_ctx_attach`, via shared helpers (`pocket_ctx_sizes_of`, `_pinned`, `_plain`, `_device`) so that the
  one-shot and the two-phase paths build the same context. The `[CTX]` counters are now mutex-guarded (profile
  runs only), and `chunk_warned` is an atomic exchange.
- `src/inference.c`:
  - the helper pool: `MYNAH_ASYNC_ADMIT=1` gives 2 threads, `=N` gives N, capped at 8;
  - a FIFO work queue and a done queue, both bounded by the slots;
  - slots in a new `starting` state that neither the step nor the retire loop touches;
  - `async_collect` attaches the device half and finishes exactly what `slot_start` does;
  - `[ADM]` profile lines (`next_job` vs start time per admission);
  - `MYNAH_ASYNC_ADMIT_INLINE` (default 8): the first N admissions of an iteration still build their context
    inline. See the reading below for why.
- Cancellation is unchanged: a `starting` slot is not polled, and a disconnect is seen at its first step. The
  disconnect test below shows nothing leaks.

**Measured before the change (`[CTX]`/`[ADM]`, warm, default path):** per admission, `next_job` (stream-writer start
and header) costs **0.11 ms**, not the ~1.5–2 ms inferred in Part 4. Context creation costs **2.2 ms**:
`ar_states` 0.82 and `codec_setup` 0.98 ms, the device part 0.30 ms. So ~90% of admission is host work, which can
move off the thread.

**mya1** (fresh server per arm, L4 24L profile). Wave 160×2: TTFB and TTFA p50/p95 in ms, and STREAM_RTF p95:

| Arm | Fresh | Warm | Server admission time per wave, fresh / warm |
|---|---|---|---|
| default | TTFB 158/739, TTFA 1177/1196, RTF 0.651 | TTFB 98/204, TTFA 600/656 | 1,123 / 573 ms |
| async (2 threads, all async) | TTFB 21/29, **TTFA 543/771**, RTF 0.863 | TTFB 24/32, **TTFA 427/475**, GOOD | 379 / 199 ms |
| async, 3 threads | TTFA 541/700 | TTFA 433/472, GOOD | 408 / 197 ms |
| async + pool prefill + width buckets | TTFB 21/26, **TTFA 406/513** | TTFA 407/459, GOOD | 233 / 188 ms |

- **Checks:**
  - md5 at batch 1, default vs async: identical.
  - Self-check: PASS.
  - 160 requests dropped after 0.3 s, then 40 dropped after 2 s: queued, active and streams all back to 0 within
    1 s, 160 disconnect lines, and the next request is served normally.
- **ttsbench `--procs 4` at c160** (aggregate / burst / steady p50/p95, underruns):
  - default: 580/595 · 591/598 · 137/160 · 5;
  - async: 218/474 · 467/1227 · **197/228** · 66;
  - async + parts 1+2: 207/1224 · 430/1226 · 181/218 · 80.
- **Steady guard, 2-min closed loop at C160, twice per arm:**
  - default: 167.0 / 168.7 audio-s/s, RTF p95 0.864 / 0.857, TTFA p95 152 / 150 ms;
  - async: 169.1 / 167.9 audio-s/s, RTF p95 0.850 / 0.851, **TTFA p95 218 / 219 ms**.

**Reading:**
- Async admission removes the serial host work from the burst: server admission time per wave drops 3×, TTFB drops
  from ~200–740 ms to ~30 ms, and warm-burst TTFA p95 goes from 656 to 475 ms (GOOD).
- It adds **one step of first-audio latency in steady state**. A request submitted to a helper during iteration k
  is collected at the top of k+1, so its prefill and first frame move one step later: TTFA p95 +67 ms, and the
  steady part of ttsbench shows the same +60 ms.
- Throughput and RTF are unchanged. The helpers also contend for the 4 vCPUs: context creation rises from 2.2 to
  3.1 ms.
- Hence `MYNAH_ASYNC_ADMIT_INLINE` (mya2): keep a burst's first admissions inline so steady state behaves exactly as
  before, send only the excess to the helpers, and collect again right before the prefill pass so the contexts that
  finished during the admission pass join the same iteration.

**Status at end of day (2026-09-30):**
- mya2 was **not run**. It was cancelled from the box queue to finish the day, and is moved to the next session.
- Its code is already in the worktree, uncommitted:
  - the `MYNAH_ASYNC_ADMIT_INLINE` threshold, default 8;
  - a non-blocking `async_collect` right before the prefill pass.

  It builds cleanly with `clang -fsyntax-only` on the Mac, but it has not been rebuilt on the box.

**Plan for mya2**, box chain `mya2.sh` (in the lab scratch; recreate it from `mya1.sh`):
1. Build and self-check. md5 at batch 1, default vs async.
2. Waves fresh + warm for default, `ASYNC_ADMIT=1` (inline 8), inline 16, and inline 8 + pool prefill + width
   buckets.
3. ttsbench `--procs 4` at c160, split into burst and steady.
4. The 2-min C160 steady guard, twice for default and twice for inline 8.

**Acceptance:**
- Steady TTFA p95 back at the default level (~150 ms).
- RTF p95 and audio-s/s within noise.
- Warm-burst TTFA p95 ≤ ~500 ms, fresh with parts 1+2 ≤ ~520 ms.

**Until then keep `MYNAH_ASYNC_ADMIT` off.** mya1 shows a clear burst gain but a +67 ms steady-state TTFA p95
regression (150 → 218 ms at C160) with all-async admission.

## Ports from the Python engine (TODO, low priority, 2026-09-30)
Wins measured on a reference PyTorch implementation of the same 24L model on the same L4 (sm_89), worth porting to the
CUDA path, each behind its own flag (default off), A/B with 60-s knees then one soak:
1. **Mimi SEANet decoder convs in bf16** (activations + weights, fp32 accumulate): step -17 % at 192 rows, SNR vs fp32
   43-45 dB at temperature 0, no audible difference. SEANet is ~49 % of the step at 160 rows, so this is the biggest lever.
2. **FlowLM weights in bf16** (tensor-core GEMMs, fp32 accumulate): RTF p95 -8/9 %, throughput +11 %. FP8 weights gave
   nothing over bf16 (per-tensor scales only) - skip.
3. **Shared voice-prefix KV**: the voice-prompt KV is computed once per voice and referenced by all rows (copy-free);
   VRAM -2 GB at 160 rows, step -10/15 %, sample-identical.
4. **First-segment split** of the text so the first chunk starts earlier under burst.
Not worth porting: 8-bit KV storage (-9/10 % end to end without a fused quantize+scatter), KV tiering to host RAM,
lead-aware row scheduling, FF/flow-head fusion (-1 %).
