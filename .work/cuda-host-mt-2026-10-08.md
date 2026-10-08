# CUDA serving loop: where the scheduler's host time goes, and what was moved off it

Branch `perf-host-mt`, 2026-10-08. Box: Vast.ai NVIDIA L4 (24 GB, 72 W cap), AMD EPYC 7702
(Zen 2, ~3.35 GHz boost), container quota ~24.6 CPUs, shared with two other jobs under one
measurement lock. Model `pocket-english-24l`, `ROW_CAP=384`, every current default on,
closed-loop ladder (`tools/gpu/knee_closed.sh`, v2 corpus, 4 voices, 4 client processes).

**Short answer.**

1. The new `MYNAH_SERVE_PROFILE=2` table (`[HOSTP]`) shows that at C320 on this L4 about **half of
   the scheduler's "host" time was not CPU work at all**: it was the host waiting for the device
   in places the sync profile does not count.
   - **9.3 ms per iteration** in the prefill tile's text upload: a pageable `cudaMemcpyAsync`
     first waits for the stream, i.e. for the decode gang queued just before the prefill pass.
     The next step was queued only after it, so the GPU also idled there.
   - **~5 ms per iteration** in admission, waiting on a parked slot set's fence
     (`mynah_backend_fence_wait`, not a counted sync). This one is real device time and is left
     as is; the profile now reports it as device wait.
2. The per-row loops are smaller than estimated on the L4, and the largest of them was waste:
   - the Mimi tile compacted each row's **stale host codec K/V** (a ~2 MB memmove per row every
     ~16 frames): 2.0 ms per iteration, 9 µs per row;
   - the rest: gang landing 0.5 ms, one-sync row staging 0.25 ms, decoder table patch 0.5 ms,
     backbone row preparation (part of 0.86 ms), PCM delivery 0.5 ms.
3. Changes (all bit-identical, all **opt-in** for now):
   - `MYNAH_CUDA_PREFILL_PINNED=1`: pinned staging, one upload, no stream wait;
   - `MYNAH_CUDA_MIMI_STALE_WINDOW=1`: the window of a device-owned Mimi row moves without
     copying its stale host K/V;
   - `MYNAH_SERVE_HOST_THREADS=N|auto`: a small persistent team runs the per-row loops.
4. Result at C256/C320 (two interleaved A/B pairs, all three on): the scheduler's own CPU time
   per iteration (sum of the top-level `[HOSTP]` host rows) **20.4 -> 8.2 ms**; the `[SERVE]`
   host line (which still counts fence waits as host) 23.6-23.9 -> 19.9-20.3 ms, host share
   52 % -> 44 %. Throughput and RTF p95 unchanged, as expected on a GPU-bound L4 (98 % SM busy
   either way); **TTFA p95 +4-6 ms** in both pairs, cause not yet separated.
5. Next target, not done: `ctx_free`'s device half costs **1.25 ms per retired request**
   (2.85 ms per iteration at C320; ~10 ms per iteration at L40S C1024 rates).

---

## 1. The profiler (`MYNAH_SERVE_PROFILE=2`)

Commit `d6e8241`. Every phase records wall time and the device wait reached inside it (stream
syncs and fence syncs, plus fence waits at level 2); host = wall - wait. Only the scheduler
thread records. Printed at shutdown after the `[SERVE]` lines:

```
[HOSTP]   phase          calls/it      wall  dev wait      host   rows/it host us/row
```

Top-level rows: `admit`, `cancel`, `dec_collect` (with `deliver` inside), `prefill`, `select`,
`step` (`step_batch`, `emit`, `gang` > `dec_submit`, `dec_land`, `post_step`), `launch`,
`retire` (`ctx_free`, `on_done`). Engine rows (`x.*`) split the decode gang (`ds_*`), the
one-sync queue (`os_*`), the prefill tile (`pt_*`), the Mimi tile (`mt_*`) and `ctx_free`
(`cf_device`, `cf_host`). An `x.*` row is inside one of the top-level rows, so the top-level host
column is what adds up to the scheduler's CPU time.

## 2. Host profile at C320, before and after

L4, C320, 90-120 s levels, ms per iteration (220-235 rows per step, ~2.4 admissions and retires
per iteration). "Before" = this branch with every flag off (`p2-off`, identical code paths to the
base); "after" = all three on, 4 threads (`ab1-new`, C256 + C320 server).

| phase | before host | after host | after device wait | note |
|---|---:|---:|---:|---|
| admit | 2.30 | 1.48 | 11.82 | the wait is the slot-set fence (device time); 0.65-0.92 ms CPU per admission |
| cancel | 0.09 | 0.08 | 0 | every 4 steps |
| dec_collect + deliver | 0.20 + 0.50 | 0.18 + 0.47 | 5.04 | delivery is 2.2 µs per row (lane mutexes); not moved |
| **prefill** | **10.26** | **1.01** | 0.00 | `pt_h2d` 9.28 -> 0.06: the pageable upload's hidden stream wait |
| step (host part) | 3.22 | 1.17 | 19.34 | |
| - dec_submit | 2.88 | 0.90 | | `mt_post` 2.04 -> 0.01 (stale window), `ds_decoder` 0.53 -> 0.46 |
| - dec_land | 0.51 | 0.21 | | on the team |
| launch | 1.04 | 0.86 | | `os_rows` 0.25 -> 0.12 on the team; `os_backbone` 0.79 unchanged |
| retire | 3.32 | 3.41 | 0.76 | `ctx_free` 2.0-2.9, all in `cf_device` (1.25 ms per request) |
| **sum of top-level host** | **20.4** | **8.2** | | |
| `[SERVE]` host line | 25.1 (p2-off) / 23.6 (ab1-base) | 19.9 | | counts fence waits as host |

Per-row costs before (µs per row, 3.35 GHz Zen 2): Mimi window advance 8.9, one-sync backbone
launch 3.6, decoder table patch 2.4, gang landing 2.3, delivery 2.2, cancellation 1.5 (every 4th
step), one-sync row staging 1.1.

## 3. What was changed

### 3a. `MYNAH_SERVE_HOST_THREADS` -- the host team (commits `0ffd759`, `e497264`)

`src/hostpool.c`: a private persistent team (not the kernel pool, which the GPU server runs at
`MYNAH_THREADS=1`).

- A region is one 64-bit word (generation, next row, row count, chunk) plus `fn`/`ud` and a done
  counter. Rows are claimed in fixed chunks (~4 per thread, at least 8 rows) by compare-and-swap;
  the caller always claims too, so a worker that wakes late claims less and nobody waits for a
  sleeping thread to start. `fn`/`ud` are read only after a successful claim and stay valid until
  the submitter has seen every claimed row finish, so a region's argument can live on the stack.
- Workers spin `MYNAH_SERVE_HOST_SPIN_US` (50) after a region, then park on a condition variable.
- Regions below `MYNAH_SERVE_HOST_MIN_ROWS` (64) rows, nested regions and a second submitter run
  inline.
- Contract per body: row r writes only row r's state and result slot; no CUDA call; anything
  order-dependent (first error reported, counters, device submissions, KV growth) is merged
  serially in row order afterwards. The result therefore cannot depend on which thread ran which
  rows.

Loops moved onto it:

| loop | where | serial merge |
|---|---|---|
| one-sync row staging: put back an unused draw, stage the previous latent, draw the noise (per-row RNG) | `pocket_onesync_step` | none needed |
| gang landing: PCM from the pinned gang rows, finite check, frame finish, leftover copy | `pocket_decode_gang_land` | non-finite rows dropped in row order (same error) |
| batched backbone row preparation: checks, positions, strides, input staging | `pocket_cuda_backbone_step_batch_impl` | early returns and KV growth in row order on the scheduler |
| Mimi tile host window advance | `pocket_cuda_mimi_tile` | the last failing row's message, as the serial loop left it |
| decoder graph table patch (also re-ordered slot-major: one slot's cells are contiguous) | `decoder_table_patch` (`backend_cuda.cu`) | none needed |

`MYNAH_SERVE_HOST_THREADS=N` (0..16, 0/1 = off) or `=auto` (off up to 8 usable CPUs, 2 up to 16,
4 above, from `mynah_usable_cpus()`). Unset = off.

On this L4 the team brought the landing 0.51 -> 0.21 ms and the row staging 0.25 -> 0.12 ms per
iteration; the backbone preparation did not move (its cost is in the launch, not in the per-row
checks), and the table patch moved little (few rows change per step thanks to L11/L22).

### 3b. `MYNAH_CUDA_PREFILL_PINNED=1` (commit `dc196a8`)

The prefill tile copied each row's text embeddings with `cudaMemcpyAsync` from the request's
malloc'd buffer. From pageable memory that call waits for the stream first; the prefill pass runs
right after the decode gang is queued, so the scheduler waited for the whole decode (12.7 ms per
call, 9.3 ms per iteration) and queued the next step only afterwards. Now the rows go into one
pinned staging buffer (owned by the model state, grown when needed) and up in one copy; a fence
recorded after the upload is waited for before the next refill (complete by then). Same bytes in
the same device buffer.

### 3c. `MYNAH_CUDA_MIMI_STALE_WINDOW=1` (commit `3d77b94`)

A row the Mimi tile owns keeps its codec K/V only in the device ring; the host window is never
read again (a failure drops the row). Advancing the host offset still compacted that window, a
~2 MB memmove every ~16 frames per row. `mynah_transformer_ar_state_prepare_window_stale` moves
the window identically without carrying the stale K/V (`window/stale-advance` in `window-test`
checks base, offsets and refusals against `_prepare_window`).

## 4. Gates

### Identity (docs/benchmarking.md protocol)

Base = `28b678a` built separately; arm = all three on, `MYNAH_SERVE_HOST_THREADS=4
MYNAH_SERVE_HOST_MIN_ROWS=2` (so 32-row CLI regions really go parallel).

| check | result | team actually used |
|---|---|---|
| CLI `--batch 32` | 32/32 identical | 250 regions in parallel |
| CLI `--batch 32 --stream` | 32/32 identical | 379 regions in parallel |
| server C1, 40 s | 95/95 identical | (1 row: inline; pinned upload and stale window exercised) |
| server C64, 40 s, saved audio | 285/1881 identical | 11,748 regions in parallel |

C64 is not a deterministic check: batch composition follows arrival timing, and the base-vs-base
C64 reference run was cancelled when the box was released (to do: run base twice at C64 for the
noise floor before reading 285/1881 either way).

### Race check

`make hostpool-test` (new, in `test-c`): 4000 regions of 1..1100 rows, parked and spinning
workers, region argument on the stack, every row compared with the serial loop and visited once.
Clean under `-fsanitize=thread` at 4 and 16 threads (macOS clang). The bodies on the engine side
touch only their row's context and per-row slots; the costmap regions they open are thread-local.

### Performance (L4, 2-minute levels, audio-s/s / RTF p95 / TTFA p95)

| run | C256 | C320 | `[SERVE]` host per iteration |
|---|---|---|---|
| ab1 base | 382 / 0.634 / 100 ms | 374 / 0.793 / 124 ms | 23.59 ms (52.2 %) |
| ab1 new (all on, 4 threads, profile 2) | 374 / 0.644 / 106 ms | 372 / 0.796 / 129 ms | 19.87 ms (44.0 %) |

| ab2 new (first in this pair) | 372 / 0.648 / 107 ms | 369 / 0.801 / 130 ms | 20.28 ms (44.2 %) |
| ab2 base | 372 / 0.649 / 103 ms | 369 / 0.804 / 126 ms | 23.91 ms (52.0 %) |

Over the two interleaved pairs: throughput and RTF p95 are unchanged (C320: 374/372 and 369/369
audio-s/s, RTF p95 within 0.003), the `[SERVE]` host time drops 23.6-23.9 -> 19.9-20.3 ms per
iteration, and **TTFA p95 is 4-6 ms higher in both pairs** (C256 +6/+4, C320 +5/+4 ms). The
cause is not identified; candidates are the team's spinning workers taking CPU from the stream
writers on this 24.6-CPU quota, or the earlier launch shifting where admissions land relative to
the step. The A/B used all three flags at once; separating them (and `MYNAH_SERVE_HOST_SPIN_US=0`)
is the first thing to run next.

The L4 is GPU-bound (97-98 % SM busy in every run), so a lower host time does not show up as
throughput here: the freed host time becomes device wait (admission fence wait 4.9 -> 11.8 ms,
decode collect 2.8 -> 5.0 ms).

## 5. Recommended defaults

All three stay opt-in until the TTFA p95 +4-6 ms seen with all three on is attributed (one flag
at a time on the L4, then a host-bound screen).

- `MYNAH_CUDA_PREFILL_PINNED`: the best default candidate once screened alone. It removes
  a hidden stream wait on the scheduler's critical path and a GPU bubble before every launch that
  follows a prefill pass (2/3 of iterations at C320); bit-identical.
- `MYNAH_CUDA_MIMI_STALE_WINDOW`: same; pure waste removed, bit-identical.
- `MYNAH_SERVE_HOST_THREADS`: keep opt-in. On the L4 it saves ~0.5 ms per iteration at ~230 rows
  and nothing shows in throughput. On a host-bound GPU at ~1000 rows the moved loops scale to
  ~2-3 ms per iteration (landing, staging, table patch); try `=auto` there.

## 6. Expected effect on host-bound GPUs (L40S class), to be measured

Scaling the per-row costs from 230 to ~1000 rows and the per-request costs to ~8 admissions and
retires per iteration (C1024, ~9 s requests), on a 3.35 GHz Zen 2-class core:

- prefill upload wait: removed. It is a wait for the decode gang, so its size follows the
  decode's GPU time; on a fast GPU it is shorter than on the L4, but it also delays the next
  launch, which is what matters when the host is the bound.
- Mimi window copies: ~9 µs x 1000 rows = ~9 ms per iteration removed. This is likely the largest
  single CPU saving at L40S scale.
- team: landing ~2 ms, row staging ~1 ms, table patch up to ~2 ms when many rows change; with 4
  threads ~3/4 of that.
- Still on the scheduler: `ctx_free` device half ~1.25 ms per retire (~10 ms per iteration at 8
  retires), admission ~0.7 ms CPU per request (~5-6 ms), delivery ~2 µs per row (~2 ms),
  one-sync backbone launch ~3.6 µs per row (~3.6 ms).

What to run on an L40S-class host: the C768-C1024 knee with `MYNAH_SERVE_PROFILE=2`, A/B of
(a) nothing, (b) `MYNAH_CUDA_PREFILL_PINNED=1 MYNAH_CUDA_MIMI_STALE_WINDOW=1`, (c) (b) +
`MYNAH_SERVE_HOST_THREADS=auto`, plus the C64 base-vs-base noise floor for the concurrency
identity check.

## 7. Next targets found by the profile

1. `ctx_free` device half, 1.25 ms per request (`x.cf_device`). Suspect: `pocket_cuda_slot_park`
   -> `mynah_backend_graph_forget_parked` scanning every decoder batch-graph entry and its row
   list (O(entries x width)) for each retired decoder. Add a probe there first.
2. Admission's fence wait (device time): the slot set taken is the one parked last, whose fence
   is the decode gang still running; preferring a set whose fence has completed
   (`mynah_backend_fence_query`) would let admission overlap the decode.
3. `os_backbone` 3.6 µs per row: the per-row preparation is not the cost (parallel made no
   difference); look at the KV table key loop and the graph launch.
4. Delivery: 2.2 µs per row in the server callback (two lane-mutex pairs per row); a per-lane
   batched hand-off would halve the locking without changing any byte.

## Flag reference

| flag | default | effect |
|---|---|---|
| `MYNAH_SERVE_PROFILE=2` | off | `[HOSTP]` per-phase host table at shutdown |
| `MYNAH_SERVE_HOST_THREADS` | off | `N` (0..16) or `auto`: team size for the per-row host loops |
| `MYNAH_SERVE_HOST_MIN_ROWS` | 64 | rows below which a region runs inline |
| `MYNAH_SERVE_HOST_SPIN_US` | 50 | worker spin after a region before parking (0..10000) |
| `MYNAH_CUDA_PREFILL_PINNED` | off | `=1`: pinned staging and one upload for the prefill tile's text |
| `MYNAH_CUDA_MIMI_STALE_WINDOW` | off | `=1`: device-owned Mimi rows skip the stale host K/V copy |
