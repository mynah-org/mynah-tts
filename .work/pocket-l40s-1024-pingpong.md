# Pocket 24L on one L40S toward C1024: ping-pong half-batches inside one engine

Board: `PLAN.md` E15-32 (follow-up of `.work/pocket-l40s-plateau.md`). Branch `pocket-l40s-plateau`. Written
2026-10-06. Design only: nothing here is coded yet.

**Goal.** One engine reaching C1024 at stream RTF p95 ≤ 0.88 on one L40S, with one admission queue, one copy of
the weights, one KV slot pool, and no extra host threads. The design has to work on a 4-vCPU host too.

**Idea.** Split the resident rows into two groups, A and B. Each group has its own step/decode pipeline and its
own scratch. While the GPU runs group A's work, the single scheduler thread does group B's host work (finish,
emit, delivery, retire, admission, prefill, launch), and then the two swap. Everything stays on **one CUDA stream**,
ordered with events. This is meant to reproduce what two engines × 320 rows showed (913 audio-s/s at 93 % GPU
busy, L14) inside one engine.

**Name.** `MYNAH_CUDA_PINGPONG=2`, item **L26**.
- This is not the dropped L16. L16 aimed at *concurrent kernels* on two streams, and the MPS result rightly killed
  that.
- L26 is *host/device overlap* on one in-order stream. The same MPS result supports it: the 93 % came from one
  engine's GPU work filling the other engine's host gaps, not from SM sharing.

---

## 1. What the measurements say (the cost model)

### 1.1 GPU work per step as a function of width

From `dmon sm` (time with at least one kernel running) and audio-s/s, the GPU time per row-frame (80 ms of audio)
is `0.08 × sm / aps`. Single engine, base build, one server, ladder `chain1 e1-768` (2026-10-05):

| C (= executed width) | aps | sm | GPU ms per step, all rows active |
|---|---|---|---|
| 384 | 723 | 58 % | 24.6 |
| 512 | 834 | 64 % | 31.4 |
| 640 | 877 | 66 % | 38.5 |

A linear fit is exact to within 0.2 ms over these three points:

```
G(W) ≈ 3.8 ms + 0.054 ms × W        (W = EXECUTED width, i.e. after width-bucket padding)
```

**Validation against the two-engine run (L14).** Two engines × 320 rows each execute at width 384, because the
default buckets are …, 256, 384, …. Per pair of steps the model gives 2 × G(384) = 49 ms for 640 row-frames, that
is 1.02 ms of GPU per audio-second. At the measured 93 % busy this predicts **912 audio-s/s**; the run measured
**913**. The model therefore also explains *why* two engines gained only ~4 % over one at 640 despite 93 % busy:
- the fixed 3.8 ms is paid twice per cycle;
- each engine pads 320 → 384 (+20 % rows).

Neither cost is inherent to splitting. The bucket padding can be removed, and the fixed cost is ~6 % at 2 × 512.

**Cross-check, single engine, all flags, C896** (the scratch capacity clamps the bucket to 896): G(896) = 52.2 ms
gives 0.73 ms of GPU per audio-second. At 74 % busy that predicts ~1,015 audio-s/s; 985 / 960 were measured. Good
to about 5 %.

### 1.2 Host work per iteration

**Measured** (all 11 flags on, `MYNAH_SERVE_PROFILE`, averaged over a C640-896 ladder):

| host | host time per iteration |
|---|---|
| 3.7 GHz host (2026-10-05) | 22-25 ms |
| 2.6 GHz host (today's reference) | ~28 ms, at ~C800-900 |
| 2.1 GHz host (3f) | 54 ms, at C832 |

**Split into a fixed and a per-row part** (from the base runs: ~21.9 ms at 384 rows, ~29-31 ms at 640): roughly
30 % fixed per iteration and 70 % per row. For the 2.6 GHz host:

```
H(W) ≈ 8.4 ms + 0.023 ms × W_active      (2.6 GHz host, all flags; ≈ 28 ms at ~850 active rows)
```

The per-phase split is **not measured**: how much of H is finish + emit + decode preparation, and how much is
delivery, retire, admission and prefill. Stage 0 measures it. It decides between this design and the simpler L13b
(section 8).

### 1.3 Mapping to the gate

The all-active convention is the same as in the fit: period = C × 0.08 / aps.

| measured | period | stream RTF p95 |
|---|---|---|
| all flags, C768 | 67.7 ms | 0.855 |
| all flags, C896 r1 | 72.8 ms | 0.836 |
| all flags, C896 r2 | 74.7 ms | 0.960 |

So **the 0.88 gate sits at a cycle of about 68-73 ms**, with tail noise. That is the target for C1024.

### 1.4 Prediction at C1024 (2.6 GHz host, executed width 1024 or 2 × 512)

| loop | GPU per cycle | host exposed (GPU idle) | cycle | GPU busy | aps | gate |
|---|---|---|---|---|---|---|
| serial (today, all flags) | G(1024) = 59.1 | ~31 (H at ~970 active) | ~90 | ~66 % | ~910 | ✗ (~1.1) |
| + L13 (hides retire, admission, cancellation) | 59.1 | ~21-26 | ~80-85 | ~72 % | ~980 | ✗ |
| + L13b (decode queued too; emit + launch preparation stay exposed) | 59.1 | ~8-12 (estimate) | ~67-71 | ~85 % | ~1,170 | borderline |
| **ping-pong, 2 × 512** | 2 × G(512) = 62.8 | ~0 while H/2 ≤ G(512) | **~66-70** at 90-95 % busy | 90-95 % | **~1,170-1,240** | ✓, small margin |

**Host budget for ping-pong at C1024.**
- The fixed part is paid per group, so host per cycle ≈ 2 × 8.4 + 0.023 × 970 ≈ 39 ms on the 2.6 GHz host, about
  19.5 ms per half.
- The GPU item per half is G(512) = 31.4 ms. The host fits with ~35 % slack.
- On the 2.1 GHz host (H about 2×) it is ~33-35 ms per half: host-bound, cycle ~70 ms, borderline. That is still
  far better than serial (~105 ms) or L13b there (emit exposed and the host total too large: ~79 ms).

**Reading.**
- Ping-pong makes the loop GPU-bound on any host of 2.6 GHz or more. C1024 then passes only if the overlap reaches
  **≥ ~88 % GPU busy**: the GPU work alone (62.8 ms) is already ~90 % of the gate budget.
- Two things matter as much as the overlap:
  - **width buckets that match half-widths** (section 5.4);
  - **zero hidden stream syncs** (section 7, risk 1).
- C960 should pass comfortably.
- These are model numbers. Stage 0 (the zero-code A/B) checks the most important one, the cost of the split.

---

## 2. Today's loop and where the GPU idles

`serve()` in `src/inference.c`, one scheduler thread, one iteration:

```
admission → async collect → cancellation → prefill pass → select (≤ max_batch rows, rotating)
→ step_live:  step_batch (one-sync AR: queue + sync #1)  → emit_batch (host)
              → stream_gang → decode_audio_batch (codec transformer sync, gang PCM sync #2) → slot_deliver
→ retire (swap-remove)
```

- Both waits are `mynah_backend_sync` = `cudaStreamSynchronize(st->stream)`. The stream is the one per backend,
  `cuda_backend_state::stream`.
- Everything else is host time with an empty queue. That is F4/F6: the GPU idles ~35 % of the time even with all
  flags on.

L13 (`MYNAH_CUDA_STEP_OVERLAP`, coded) queues AR k+1 (`step_launch`) before retire, admission and cancellation.
Emit, decode preparation, delivery and prefill stay exposed, and an admission joins one step late. On the L4 that
cost +50-60 ms of TTFA p95.

---

## 3. The ping-pong schedule

### 3.1 One stream, two items in flight

Each group X ∈ {A, B} owns one **GPU item** per cycle:

```
item X(k+1) = [ DEC_X(k)  →  fence dX  →  AR_X(k+1)  →  fence aX ]
```

- DEC_X(k) is the gang decode of the frames that AR_X(k) produced: codec transformer, decoder graph, PCM gather and
  D2H into X's pinned PCM.
- AR_X(k+1) is today's one-sync chain, queued by `step_launch`: condition, backbone graph, EOS, lazy-hidden probe,
  flow graph and D2H.
- AR k+1 needs only emit k (host), not decode k, so both can go in the same item.

Steady state, on the GPU and on the scheduler thread:

```
GPU:   | DEC_A AR_A | DEC_B AR_B | DEC_A AR_A | DEC_B AR_B | ...
host:  | main_B ... deliver_A | main_A ... deliver_B | main_B ... deliver_A | ...
```

**main_X** starts when fence aX fires, while item Y runs:
1. Finish AR_X(k): the existing FINISH phase of `pocket_onesync_step` (commit, EOS logits, flow finite check,
   lazy-hidden bookkeeping), waiting on **fence aX instead of a stream sync**.
2. Emit X(k): `emit_batch` (latents, EOS / budget / re-prepare decisions, the noise put-back when there is no
   latent).
3. Retire X: the rows that ended at k-1 and whose last frame was delivered at deliver_X. Then cancellation for X's
   rows (L7 cadence), admission (global, assigned to a group, section 4.3) and async collect.
4. The prefill pass for X's preparing rows, on X's scratch. It runs **last**, see section 4.5.
5. Launch: decode preparation, then queue DEC_X(k) and fence dX, then `step_launch` AR_X(k+1) and fence aX. With
   that, item X(k+2) is queued behind item Y.

**deliver_Y** runs right after main_X: wait on fence dY, which normally has already fired because DEC_Y is the first
half of item Y. Then the decode collect: finite scan, PCM placement, `decode_frame_finish`, then `slot_deliver`
(L19 hand-off). AR_Y is still running meanwhile.

**The invariant that makes it safe.**
- The host touches a group's scratch, its pinned staging, its decode workspace and its rows' contexts **only while
  that group's item is not in flight**. This is the same "separation in time" argument as in L13, now per group.
- The two exceptions are both safe:
  - deliver_Y reads Y's PCM after fence dY, while AR_Y runs. AR does not touch the PCM or decode state.
  - Admission and prefill may create contexts for either group, but never step or decode them outside their own
    main phase.

**Cycle time.**

```
T ≈ Σ over the two halves of  max(host_main_X + host_deliver_Y, G_item_Y)
```

If each half's host time is at most the other group's item, then T = G_A + G_B: the GPU is never idle except for
fence-to-launch jitter in the µs range.

### 3.2 Why one stream with events, and not two streams

- **The kernels serialize either way.** MPS showed no concurrency headroom (L14), so a second stream buys no SM
  overlap.
- **One in-order stream makes every backend-wide device workspace safe for free.** Two streams would race on all of
  them:
  - `dev_scratch`, `dev_q8_activation`, `dev_bf16_activation`;
  - the cuBLAS/Lt workspaces;
  - `dec_cols` / `dec_out`, the `solo_*` buffers, the codec gang `up_proj` / `pcm_dev`;
  - the attention metadata tables `dev_batch_*`.
  Each would then need duplicating per stream, plus a cuBLAS handle per stream.
- **Only host-side pinned buffers can race** (section 5.2), and those are a short list.
- **Waits must be event waits.** With two items queued, a stream sync waits for *both*. Every steady-state wait has
  to be a fence wait on the group's own fence.

### 3.3 Latency of a frame

From the end of AR_X(k) on the GPU:
1. main_X (host);
2. the remainder of item Y on the GPU;
3. DEC_X(k);
4. deliver_X.

That is about one item plus one decode, **~40-50 ms** at 2 × 512 (item ≈ 31 ms). In the serial loop, delivery
comes ~15-25 ms after AR.

The first frame of every stream is therefore ~25 ms later than serial. But admission and prefill now run twice per
cycle (every ~32 ms instead of every 70-90 ms), which takes back roughly half an iteration of admission wait.

**Expected TTFA p95:** about flag-off +0-30 ms. That is better than L13 (+50-60 ms on the L4), because nothing
joins "one step late" here: an admission into X joins X's very next launch.

The same latency applies to every later frame, as a constant offset. It does not affect the frame *period*, which
is what stream RTF and stalls measure.

---

## 4. Driver design (`src/inference.c`)

### 4.1 Data structures

```c
typedef struct {
    int id;                                   /* 0 = A, 1 = B */
    mynah_engine_scratch *scratch;            /* own scratch, capacity = group_cap */
    size_t cap;                               /* ceil(max_batch / 2) */
    size_t rows[MYNAH_GRAPH_MAX_JOBS];        /* member slot indices, group-local order */
    size_t count;                             /* members: active + preparing + finishing */
    size_t step_rr;                           /* the rotation cursor of today's select, per group */
    /* in flight (valid while `inflight`) */
    int inflight;
    size_t ar_count;  size_t ar_pos[...];     /* group positions of AR(k+1) rows, in order */
    size_t dec_count; size_t dec_pos[...];    /* gang members of DEC(k), with first/want/pcm/produced/failed */
    mynah_engine_step_result results[...];    /* emit results of the frame being decoded */
    /* counters */
    unsigned long long items, serial_phases, fence_wait_ns, host_main_ns, host_deliver_ns;
} pp_group;
```

- **Heap allocation.** Two groups at ROW_CAP 1024 are ~150 KB. They are `calloc`'d at serve start, not put on the
  scheduler stack, which already holds the 1024-row arrays.
- **New slot fields.** `synth_slot` gets `int group` (-1 = none) and `size_t gpos` (its index in `group->rows`).
- **Swap-remove.** The global `slots[]` array stays dense (`compact_rows`). Every global swap-remove updates the
  moved slot's `group->rows[gpos]`.
- **Positions, not slot indices.** In-flight arrays hold group positions, and a group's positions change only in
  its own main phase. The other group's retire can move global slot indices freely. This replaces L13's
  `ahead` / `ahead_pos` bookkeeping and its retire simulation.

### 4.2 Group-local order (needed for the split-identity test)

Within a group, the row order evolves exactly as a stand-alone `serve()` over those rows would:
- append at admission;
- group-local swap-remove at retire;
- the same `step_rr` rotation in the select.

Each group is therefore *numerically* a stand-alone engine run, with the same widths, the same row order and the
same decode gangs. Section 6 relies on this.

### 4.3 Admission: which group

`admit_pass` stays global (the one sink, the one queue). A new slot is assigned a group at admission:

1. **Below `MYNAH_CUDA_PINGPONG_MIN` active + preparing rows** (default 128): group A. A lone non-empty group runs
   the serial path (section 4.6), so low concurrency keeps today's latency and loses nothing to the split.
2. **Otherwise, block-balanced per pass.** With `n` new rows in this pass and `X` the group whose main phase is
   running:
   - give the first `clamp(ceil((n + |Y| − |X|) / 2), 0, n)` rows to X, the rest to Y;
   - respect `cap`; a group at `cap` gets nothing.

   With n = 1-3 (steady state) that is "the smaller group, ties to the current one". The current one launches next,
   so the row joins without waiting half a cycle. A burst into empty groups gives contiguous halves.
3. **Rows never migrate between groups.** As load drops, the smaller group drains naturally. When one group is
   empty, the other runs serially.

### 4.4 Retire, cancel and fail, per group

- **Retire** (in main_X, X's rows only).
  - A row that ended at emit k still has frame k inside DEC_X(k). It is marked `finishing` and retired in main_X
    of the next cycle, after deliver_X has delivered that last frame.
  - The KV slot is held one extra item (~31 ms). At ~5 completions per iteration that is ~5 rows of capacity.
- **Cancellation** (`sink->cancelled`, every N cycles under L7) is checked only for X's rows in main_X, so the cost
  per cycle stays the same.
  - X has nothing in flight at that point. The row is dropped immediately, with today's semantics.
  - Unlike L13, no row "rides along".
- **Failure isolation** (`step_isolate`): a launch-side or finish-side failure makes main_X fall back to serial for
  that phase.
  - It drains the stream: correct, slow, rare.
  - It is counted in `serial_phases{reason=failure}`.

### 4.5 Prefill

The prefill pass for X's preparing rows (`slots_prefill_slice` restricted to `group->rows`) uses **X's scratch**,
which is idle in main_X. The batched prefill also uses the scratch's backbone workspace.

- **Position.** It runs *last* in main_X, just before the launch, for two reasons:
  - Its internal stream syncs wait for item Y, which is queued ahead on the stream. The host only waits there, but
    after the prefill the GPU idles until the launch. Running prefill last shrinks that idle time to the launch
    preparation (~1-3 ms), and only in phases that have a slice.
  - A row whose prefill completes joins AR_X(k+1), the same as flag-off.
- **Group-restricted, not global**, so that prefill batch composition per group equals a stand-alone run (section
  6). The cost is that a row assigned to Y waits for main_Y to prefill. Assigning to the current group (4.3) makes
  this rare.
- **Late admission** (`wait_arrival`) runs once per main phase. `MYNAH_CUDA_FAST_FIRST_CHUNK_WAIT_US` must stay 0:
  a deliberate wait in main_X is exposed host time.

### 4.6 When a group runs serially instead of pipelined

main_X falls back to today's `step_live` for X: synchronous step, emit, decode and delivery. The reason is counted
in `[SERVE] pingpong` each time. Cases:

- **Width.** The other group is empty, or X's width is < 2: `step_launch` refuses below 2. C1 and low C land here,
  so their audio is bit-identical.
- **Launch refused.** `step_launch` or `decode_launch` returns 1: a row that needs KV growth, an exhausted budget, a
  custom noise function, one-sync off, a decode row that is not device-resident, or a host mirror.
- **Failure isolation**, as in 4.4.

**KV growth.** It would otherwise be the most frequent refusal:
- At C1024, rows cross the initial 256-step capacity at about rows / 20 s, i.e. ~3 per cycle.
- Without VMM the growth is a copy plus a **stream sync** (`pocket_cuda_backbone_reserve`).
- Requirement: `MYNAH_CUDA_KV_VMM=1`, where growth maps a chunk with no copy and no sync, plus a **pre-grow in
  main_X**: reserve for rows within a few positions of capacity before the launch. Then launches are never refused
  for growth.

### 4.7 Profile

```
[SERVE] pingpong (MYNAH_CUDA_PINGPONG=2): cycles N, items A/B, serial phases {width, refused, failure},
        group rows mean/max A/B, host main/deliver per half (ms), fence wait (ms), stream syncs in pipelined phases N
```

**Fence waits as device wait.** They must be counted in the existing "device wait" line. Today
`mynah_cuda_fence_wait` is not a profiled sync site, so add it as one, so that host versus device-wait stays
comparable with the earlier runs.

**"Stream syncs in pipelined phases" must be ~0 in steady state.** Any non-zero site is a hidden serialization
(section 7, risk 1). The per-site table already shows it.

---

## 5. Engine and backend changes

### 5.1 Engine hooks (`src/tts_engine.h`, `src/engine_pocket.c`)

| hook | status | what changes |
|---|---|---|
| `step_launch(ctxs, n, scratch)` | exists (L13) | at the end, record a fence `aX` on the scratch (reusable per-scratch event, not a create per call) |
| `step_batch` finish phase | exists (L13, `POCKET_ONESYNC_FINISH`) | `frame_sync` waits on the scratch's fence when one is armed, instead of `mynah_backend_sync`. Byte-identical for L13 (nothing else is queued there), required here |
| `decode_launch(ctxs, n, first, frames, scratch)` | **new** (this is L13b's queue half) | everything `pocket_decode_audio_batch` queues, plus fence `dX`; returns 1 (nothing queued) when not eligible |
| `decode_collect(ctxs, n, first, frames, out, out_count, failed, scratch, err)` | **new** | wait `dX`, then today's post-sync half: finite scan, PCM placement (L20 lending), `pocket_decode_frame_finish`, `decoded_frames` |

**Splitting the decode is the main engine work.**
- Today `pocket_decode_audio_batch` has a stream sync after the codec transformer (`mynah_backend_sync` at the end
  of the codec stage, ~line 12810) and another after the gang PCM gather (~line 13215).
- The first exists for the host mirror copy and the window-offset commit. With `device_handoff` and no host mirror,
  the copy is not needed, and the offset commit is host bookkeeping that needs no device result. So it moves to the
  launch, and that sync disappears.
- Whenever a host mirror or a per-row CPU fallback would be needed, `decode_launch` refuses (serial phase).
- **A shared prerequisite.** The same split is L13b. Do it first (section 8): it is useful with or without groups.

The safety net (`pocket_cuda_ahead_discard`) is per scratch already, so a call on scratch A never discards B's
queued frame. Extend it to an armed decode on the same scratch.

### 5.2 Backend: host-pinned state that must be per group ("lane")

These are backend-wide today, live in `cuda_backend_state`, and are written by the host between two GPU uses:

| state | today | hazard with two items in flight | fix |
|---|---|---|---|
| `codec_gang.pcm_host` / `pcm_dev` (gang PCM gather target) | one per backend | DEC_Y's D2H overwrites the PCM while deliver_X reads it. **Silent audio corruption** | `codec_gang[lane]` |
| `codec_gang.pcm_meta_*`, `up_meta_*` (+ events) | one per backend, event-guarded | correct, but X's launch blocks on `cudaEventSynchronize(meta_event)` recorded by Y's queued decode | `codec_gang[lane]` |
| decoder batch graph entries reused by shape (`find_decoder_batch_graph_shape`, L11 table patch) | keyed by width only, guarded by `entry->done` | both groups at the same gang width share one entry. X's patch waits for Y's decode to finish: serialization, not a race | add the lane to the key (≤ 2 entries per hot width) |
| `active_decoder_tables`, `active_decoder_upload_slot` | single | only during a capture / record, which is synchronous within one launch call | none; keep the capture inside one call |
| tile workspace `meta_host` + event | single, event-guarded | prefill tiles of X could wait for a tile op of Y | in practice no tile op is ever inside an item; check the per-site table |
| `host_buf` (mapped staging) | single | used only by the synchronous host matmul paths, not by the resident serving path | none |

**Lane selection.** The engine passes its scratch as the lane: `mynah_backend_set_lane(backend, scratch_lane)` at
the top of each scratch-bound entry. A plain field is enough, since one thread does all of it. Lane 0 is today's
state, so flag-off is unchanged.

**Pipeline graphs need nothing.** `cuda_pipeline_graph_entry` is already keyed by `(key, identity = scratch)`, so
two scratches get two backbone/flow graph sets automatically.

### 5.3 Graphs, captures and their memory

- **Backbone / flow / condition / lazy graphs.**
  - Today: one set per bucket up to 1024 for one scratch, 19 buckets.
  - Ping-pong: two sets of buckets up to 512 (~16 each), about 1.7× the count.
  - A 24-layer graph exec is a few hundred kernel nodes; its device footprint is in the MB range. Estimated total
    increase: **< 150 MB**.
  - Capture is legal with work pending on the stream: relaxed mode, and the stream is a created stream, not the
    legacy default. A capture inside main_X is a one-off host stall per bucket and lane.
  - Pre-capture both lanes' buckets at warm-up if the start-up budget allows.
- **Decoder batch graphs** (exact width per gang; L25 is still open).
  - Each entry's tables are `upload_slots × 4 × batch × 8 B`, on pinned host plus device. At ping-pong widths
    (~400-520) each entry is about half the size of today's (~800-1000).
  - The number of distinct widths seen is similar. Net memory: neutral to lower.
  - `CUDA_DECODER_GRAPH_CAP = ROW_CAP` still bounds the cache.

### 5.4 Width buckets

- **The problem.** The default list jumps 384 → 512 → 640, which at half-widths pads a lot. At C896 (~850 active,
  ~425 per group) each group pads to 512, +20 %. That is the same loss that made the two-engine run inefficient.
- **The fix.** The scratch capacity per group is `cap`, and a step executes at `min(bucket, cap)`. Add half-width
  buckets for ping-pong: 320, 448, 576 (and 704, 896 for serial).
  - Every A/B arm on the box uses the same explicit `MYNAH_CUDA_WIDTH_BUCKETS` list, so the comparison is fair.
  - Cost: 2-3 extra graphs per lane.

### 5.5 VRAM

| item | delta vs one engine today |
|---|---|
| weights, KV slot pool, voice cache | 0 (one copy) |
| scratch activations (2 × 512 vs 1 × 1024) | ~0 (the per-row size is the same: ~45 KB/row for the backbone slabs) |
| codec gang workspace × 2 (PCM ~4 MB per 512 rows) | +~10 MB |
| pipeline graph sets × 2 | +50-150 MB |
| decoder graphs | ~0 to negative |
| **total** | **+0.1-0.3 GB** |

For comparison, two engines × 320 used 33.5 GB at ready, against 24.8-30.5 GB for one engine.

---

## 6. Determinism and audio identity

- **RNG.** Every draw comes from the row's own RNG: `ctx->rng`, the early one-sync draw saved in `onesync_rng`,
  and the put-back on discard or no latent. Group membership never changes a row's sequence of draws, and seeds are
  unchanged.
- **The only numerical difference is width.**
  - cuBLAS/Lt choose the algorithm per shape, and TF32/bf16 GEMMs are not batch-invariant.
  - A row stepped at width 487 can differ in the last bits from the same row at width 974.
  - This is the "inv" class of section 3b of the plateau note. Expected, and the same as any change in arrival
    timing.

| check | expectation | why |
|---|---|---|
| server C1, `pocket_ladder` fixed seed, `PINGPONG=2` vs off | **id**, all files | one row → group A, B empty → serial phase; the profile must show `pipelined items = 0` |
| CLI `--synthesize --batch 32` seeds 1000-1031, default MIN (128) | **id** 32/32 | below MIN everything is in group A, serial: proves the flag-off path is unchanged |
| **split identity**: CLI `--batch 32 --seed 1000` with `PINGPONG=2 PINGPONG_MIN=2`, vs flag-off CLI `--batch 16 --seed 1000` and `--batch 16 --seed 1016` | **id** 32/32 (requests 0-15 vs run 1, 16-31 vs run 2) | the block assignment puts 0-15 in A and 16-31 in B. Each group has a stand-alone engine's widths, row order, prefill composition and decode gangs (4.2, 4.5). This is the test that exercises the pipelined path; it must show `pipelined items > 0` and `discarded = 0` |
| server C64 / C256 under concurrency | **inv**: same length per request, within self-check tolerance | composition follows timing, as in every concurrent run |

The split-identity test is the strongest of these. It proves that pipelining changed *nothing* but scheduling:
fences, lanes, retire timing, the deferred last-frame delivery and per-lane graphs all included. A race on a shared
pinned buffer (5.2) shows up there as a mismatch.

---

## 7. Composition with the existing flags

| flag | with ping-pong |
|---|---|
| **L13** `STEP_OVERLAP` | **subsumed.** Ping-pong uses its queue/finish split (`step_launch`, the FINISH phase, the discard net) but not its driver logic (ahead marks, retire simulation, late join). With `PINGPONG=2`, L13 is ignored, with a start-up line |
| ONE_SYNC (default on) | required; it is the AR item |
| **L6** deferred release | **required.** Without it every `ctx_free` drains the stream and waits for the other group's item. With it a parked context is fenced, and a re-take waits on that fence (rare) |
| L7 cancel every N | per cycle, over the phase group's rows only, so half the cost per phase |
| L8 writev | unaffected (writer threads) |
| L10 dup-check epoch | per `step_batch` / `step_launch` pre-flight per group; unaffected |
| **L11** decoder table patch | needs the lane in the shape key (5.2), otherwise the `entry->done` wait serializes the groups |
| L12 survivor flow reuse | per-scratch flow layout; unaffected |
| L19 delivery threads | unaffected. A stream lives in one group, so its PCM order is preserved. On 4 vCPU use 1-2 helpers (section 9) |
| L20 PCM direct | per-context lent buffers, consumed in deliver_X before X's next decode. The gang PCM source moves to `codec_gang[lane]` |
| L21 lazy hidden | per scratch (`cuda_hidden_lazy_scratch`); emit in main_X settles it before the launch, as in L13 |
| L22 KV table cache | per-scratch keys. Groups are stable, so cache hits rise (less row churn per scratch) |
| L24 validate once | per decoder; unaffected |
| async admission | unaffected (`ctx_attach` takes no scratch) |
| KV VMM | strongly recommended (4.6) |
| decoder lane, parity dump, `MYNAH_PREFILL_SLICE=0` | refused, with a start-up line (same as L13) |

**Why groups and not more depth in one group.** Inside one group, emit k is on the critical path between AR k and
AR k+1: L13b still leaves emit plus the launch preparation exposed. Ping-pong hides it behind the *other* group's
item. Section 8 turns that into a decision rule.

---

## 8. Ping-pong or L13b? A decision gate after Stage 0

Both need the decode split (5.1). They differ in what stays exposed and what they cost:

| | L13b (one group: decode queued, AR k+1 queued after emit k) | ping-pong (two groups) |
|---|---|---|
| exposed host per cycle | emit + decode preparation + launch preparation (`e`) | ~0 while H/2 ≤ G(W/2) |
| extra GPU work | 0 | +3.8 ms per cycle (the second fixed cost), plus bucket padding |
| C1024 cycle on the 2.6 GHz host | 59.1 + e | ~62.8 / busy |
| C1024 cycle on a 2.1 GHz host | max(59.1 + 2e, H ≈ 70) ≈ 79 | max(62.8, ~70) ≈ 70 |
| TTFA | +1 step (an admission joins after the queued step) | about +0-30 ms (3.3) |
| code | small on top of L13 | groups, lanes, scratch × 2 |

**Rule.**
- If Stage 0 measures `e` (finish + emit + decode preparation + launch preparation, per iteration) **≤ ~5 ms** on
  the 2.6 GHz host, L13b reaches about the same cycle with less code. Do L13b, and keep ping-pong for slow hosts.
- If `e ≥ ~6 ms`, which the item estimates suggest (emit includes the Box-Muller draws of rows × 32, latent
  copies, EOS and flow scans; decode preparation includes table patches), ping-pong wins at C1024 by ~5-10 % of
  cycle. It is also the only option that stays under the gate on a ~2 GHz host.

---

## 9. CPU-core budget (no new threads)

- **The scheduler.**
  - Today it waits in `cudaStreamSynchronize` with the default `cudaDeviceScheduleAuto`, which normally spins. One
    core is 100 % busy either way: ~35-55 % real work, the rest spinning.
  - Ping-pong turns that spin into work. At C1024 on the 2.6 GHz host: ~39 ms of host work per ~66 ms cycle ≈ 60 %
    of one core, the rest fence waits.
  - **No helper thread is added**, and the overlap comes entirely from reordering work on that one thread.
- **What does grow is per-audio-second work on threads that already exist:**
  - the HTTP writers (L8);
  - the delivery helpers (L19);
  - async admission helpers, if on.

  It grows with throughput: +25-35 %, the same as any throughput win.
- **4-vCPU host (2 cores + SMT).**
  - The scheduler should own one physical core: pin it, and keep its SMT sibling lightly loaded. An SMT sibling
    running writer threads costs the single-thread speed ~20-30 %, and with it the host-per-half margin (~35 %
    slack on the 2.6 GHz host).
  - `MYNAH_STREAM_DELIVER_THREADS` 1-2, not 4.
  - Test `MYNAH_CUDA_SYNC=blocking` (or `yield`). It adds ~10-50 µs of wake-up per fence wait, ~4 waits per cycle,
    which is negligible against ~66 ms. In exchange the scheduler stops spinning while it waits, and the other
    vCPUs get that time back.
  - The load generator must not share this host in the 4-vCPU test (the 2026-10-04 run did: client at 89-105 %).
- **Fewer groups is the safe direction.** With N > 2 groups the fixed host and GPU costs grow linearly, and nothing
  is gained once H/2 ≤ G(W/2). So **N = 2 only**; `PINGPONG=3` and up is refused. If the host total exceeds the GPU
  total, no grouping helps: the single thread is the bound, and only host diet does.

---

## 10. Risks

1. **Hidden stream syncs (the main one).** Any `mynah_backend_sync` or `cudaStreamSynchronize` reached in main_X
   waits for item Y and silently gives back the overlap of that phase. Candidates:
   - KV growth without VMM;
   - `codec_gang_grow_pair`, or the `up_proj` grow;
   - slot-pool warm-up or re-take;
   - prefill slices (bounded by 4.5);
   - step isolation;
   - decoder close, graph forget, and the L6-off drains.

   Mitigation:
   - the "stream syncs in pipelined phases" counter, with the call site;
   - the per-site table must be fence-only in steady state;
   - fix any recurring site before measuring speed.
2. **A shared pinned-buffer race** (5.2): corrupted audio, not a crash. Mitigation:
   - the split-identity test;
   - `MYNAH_CUDA_PINGPONG_CHECK=1` (debug): a full stream drain at every phase boundary. It must produce
     byte-identical audio to `PINGPONG=2` on the same schedule; a difference means a race.
3. **The split costs more GPU than modelled.** Two × G(512) against G(1024) is +6 % on the fit. If Stage 0 shows
   ≥ +10 %, the C1024 ceiling drops below the gate even at 95 % busy.
4. **Bucket padding at half-widths** (5.4). It cost the two-engine run ~20 %. Use the explicit half-width list.
5. **First-frame latency** +~25 ms (3.3). Watch TTFA p95. If needed, a four-item variant
   (AR_A | DEC_B | AR_B | DEC_A) delivers right after each decode, at the price of tighter host/GPU pairing per
   quarter.
6. **Group imbalance.** Burst retirements can skew A/B by tens of rows. That costs 0.054 ms per row of difference,
   and admission corrects it within a few cycles. A group at `cap` refuses admission while the other has room:
   counted, and should be ~0.
7. **Slow hosts.** On ~2 GHz cores the host per half exceeds the GPU item. Ping-pong is still the best option there
   (~70 ms against ~79-105 ms), but C1024 is not guaranteed.
8. **Graph captures inside a phase** (new bucket × lane) are one-off host stalls. Pre-capture at warm-up.
9. **L4 (GPU-bound).** Splitting adds ~3.8 ms of GPU per cycle and hides host time the L4 barely has (13-15 ms
   against ~60+ ms of GPU). Expect a ~1-3 % loss at C288-320.
   - `MYNAH_CUDA_PINGPONG_MIN` does not help there, since C288 > 128. So it stays opt-in per deployment, and is
     never a default on GPU-bound boxes.
   - Close-out rule (3c): a WIN on the L40S plus no regression on the L4 is required for a default. The likely
     outcome is "opt-in, documented for host-bound GPUs".

---

## 11. Staged plan (smallest proof first)

| stage | content | proves | expected |
|---|---|---|---|
| **0a** (no code) | Server `--max-batch 512 --max-inflight 1024` (today's rotation steps 2 microbatches of 512) vs `--max-batch 1024`, at C1024, all flags, same buckets. Read `dmon sm` and aps | the **GPU cost of splitting**: GPU ms per audio-s at 512-wide against 1024-wide steps. Model: +6 %. Kill criterion: ≥ +10 % | data only |
| **0b** (profile only) | Host phase timers in `[SERVE]` (finish, emit, decode preparation, delivery, retire, cancel, admission, prefill, launch preparation), plus cudaEvent device timing of AR and DEC per width. Fix `knee.sh`'s grep, which hides B896/B1024 and most of the sync sites | `e` for the decision gate (8); the AR:DEC split of G | none (`id`) |
| **1** | Decode split (`decode_launch` / `decode_collect`), driven as **L13b** on one group. Fence-based finish | removing the codec-transformer sync; the decode in flight while the host works; L13b's cycle | +10-20 % at C896-1024 (with L13) |
| **2** | Groups + two scratches + `codec_gang[lane]` + lane-keyed decoder graphs + block-balanced admission + group-local retire/cancel/prefill + `PINGPONG_MIN`. Profile line | the **overlap**: GPU busy ≥ 88 %, stream syncs in pipelined phases ≈ 0, split identity 32/32 | C1024 at RTF p95 ≤ 0.88, ~1,170-1,240 aps (if 8 says `e ≥ 6 ms`) |
| **3** | Robustness: KV pre-grow with VMM, warm-up pre-capture per lane, half-width buckets, `PINGPONG_CHECK` debug mode, serial-phase reasons | no refusal storms; tail (worst step) under control | p95/p99 gaps |
| **4** | Close-out: L4 knee, 4-vCPU check, 10-minute soak at C1024, docs (`docs/cuda-serving.md` section 7, `configs/perf/`) | 3c rule | opt-in or default |

**The smallest step that proves the overlap.**
- Stage 2 cannot be meaningfully cut further: a two-group loop with a synchronous decode would wait for the other
  group's AR inside every decode, which is exactly the serialization this design removes.
- What can be proved first, cheaply:
  - Stage 0a: the cost side of the bet, with zero code;
  - Stage 1: the decode split, which the design needs anyway and which is useful alone.
- If Stage 1's L13b already meets the gate (rule 8), Stage 2 becomes the "slow host" option, not the C1024 path.

---

## 12. Box test plan

**Box.**
- One L40S, a host of ≥ 2.6 GHz, single NUMA node or the server pinned to the GPU's node.
- `make cuda cuda-server CUDA_ARCH=sm_89 ROW_CAP=1024`.
- Server: `--max-batch 1024 --max-inflight 1024`.
- Clients on separate cores, `--client-procs 4`.
- Every job time-boxed, and a stop also kills its `knee.sh`.

**Common environment for every arm.**
- The 11 flags ("ALL").
- `MYNAH_CUDA_KV_VMM=1`.
- The same explicit `MYNAH_CUDA_WIDTH_BUCKETS=1,2,4,8,16,24,32,48,64,96,128,160,192,256,320,384,448,512,576,640,704,768,896,1024`.
- `MYNAH_SERVE_PROFILE=1`.

**Identity** (each arm prints its `[SERVE] pingpong` line):
1. CLI `--batch 32 --seed 1000`: ALL vs ALL + `PINGPONG=2`. Expect 32/32, `pipelined items = 0` (below MIN).
2. Split identity: ALL + `PINGPONG=2 PINGPONG_MIN=2`, CLI `--batch 32 --seed 1000`, against ALL CLI
   `--batch 16 --seed 1000` and `--batch 16 --seed 1016`. Expect 32/32, `pipelined items > 0`, no "discarded"
   line, stream syncs in pipelined phases = 0.
3. Same as 2 with `PINGPONG_CHECK=1`: byte-identical to 2.
4. Server C1 (`pocket_ladder`, fixed seed): ALL vs ALL + `PINGPONG=2`. Expect all identical.
5. Server C64: equal per-request lengths and the self-check passing.

**Speed** (2-minute levels C768 → C896 → C1024, then C1152 if C1024 passes; two repetitions, arm order
alternated):

| arm | env |
|---|---|
| ALL | the 11 flags |
| ALL + L13 | `MYNAH_CUDA_STEP_OVERLAP=1` |
| ALL + L13b | Stage 1 |
| ALL + PP | `MYNAH_CUDA_PINGPONG=2` (Stage 2) |
| 0a | ALL, `--max-batch 512 --max-inflight 1024` (C1024 only) |

**Metrics per level.**
- Throughput and quality of service: audio-s/s, RTF p95, TTFA p95, gap p95, stalls@250, failures.
- GPU: `dmon sm` and W.
- Host per cycle, and host main/deliver per half.
- Device wait, split into fence and stream.
- Syncs per cycle by site.
- Ping-pong counters:
  - pipelined vs serial phases, with the reason;
  - group rows mean/max;
  - fence wait time.
- Graphs: captures per lane, decoder re-records and patches per lane.
- VRAM at ready and at the end.
- Server CPU (the scheduler thread and the total).

**Pass.**
- C1024 RTF p95 ≤ 0.88 in both repetitions, with 0 stalls and 0 failures.
- `dmon sm` ≥ 88 %.
- TTFA p95 ≤ ALL + 40 ms.
- VRAM delta ≤ 0.5 GB.
- Stream syncs in pipelined phases ≈ 0.

**4-vCPU emulation.**
- Server pinned to 4 vCPUs (2 cores + SMT siblings, `PIN=`), clients on other cores.
- ALL vs ALL + PP at C640 and C768, with `MYNAH_STREAM_DELIVER_THREADS=1` and 2, and `MYNAH_CUDA_SYNC` at
  default and at `blocking`.
- Expect a gain and no failures. Report server CPU per thread.

**Then.**
- A 10-minute soak at C1024 (ALL + PP).
- The L4 knee C256 / C288 / C320 (ALL vs ALL + PP) for the 3c rule.

**Watch.**
- The one-time "step queued ahead was discarded" line must never appear.
- Any `serial phases{refused}` rate above ~1 % of phases points to KV growth or a decode-launch refusal.
- A rising "fence wait" with GPU busy < 85 % means the host per half exceeds the item: host-bound. Compare it with
  the phase timers.
