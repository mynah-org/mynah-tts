# Pocket 24L on one L40S toward C1024: L13 + L13b, the AR step and the decode both queued

Board: `PLAN.md` E15-32 (follow-up of `.work/pocket-l40s-plateau.md`, "L13 design"). Branch `pocket-l40s-plateau`.
Written 2026-10-06 from a code read only. Nothing here is coded or measured. Every number marked "model" comes from the
cost model in section 2 and must be replaced by Stage 0 measurements.

Sibling notes, same day: `.work/pocket-l40s-1024-pingpong.md` (two row groups in one engine, L26) and
`.work/pocket-l40s-1024-host-profile.md` (where the ~28 ms of host time goes).

---

## 0. Summary

**Question.** L13 queues AR k+1 before retire, admission and cancellation, but the gang decode of step k stays
synchronous. Can the decode of step k be queued too, so that the host does emit, delivery, retire and admission
while the GPU runs decode k and AR k+1? What does that do to throughput, TTFA and the gap between chunks?

**Answer.**

1. **Two ways to queue both.** They differ in where decode k sits on the one stream:
   - **Decode-ahead (recommended L13b).** Decode k is queued before AR k+1 and collected by event in the **same**
     iteration.
   - **Decode-behind (the "one iteration later" design).** Decode k is queued after AR k+1. Its PCM is collected in
     the **next** iteration from a double-buffered pinned buffer.
2. **Throughput: they are within ~4 % of each other.** Both make the loop GPU-bound on the 2.6 GHz host: H ≈ 28 ms
   against G ≈ 34 ms of GPU work per iteration at ~C896.
   - Decode-ahead leaves only `e` exposed: finish + emit + the head of the decode submission, ~2-3 ms.
   - L13c, which queues the prefill of rows admitted after the launch right behind the queued AR step, covers that
     too whenever admissions exist. At C ≥ 768 they always do.
   - Model at ~C896: iteration 52 → ~35-36 ms, **+44-49 %**. L13 alone: ~42 ms, +23 %.
3. **TTFA: decode-behind costs a whole extra iteration.** Decode-ahead does not.

   | model, ~C896, mean | iteration | TTFA closed loop | TTFA Poisson | steady gap |
   |---|---|---|---|---|
   | all 11 flags | 52 ms | ~50 ms | ~58 ms | 52 ms |
   | + L13 | ~42 ms | ~80 ms (+30) | ~63 ms | 42 ms |
   | + L13 + L13b (decode-ahead) | ~36 ms | **~50 ms (±0)** | ~67 ms (+9) | 36 ms |
   | + L13c (prefill-behind) | ~35 ms | ~48 ms | ~65 ms | 35 ms |
   | + L13d (first frame first) | ~37 ms | **~35 ms (−15)** | **~52 ms (−6)** | 37 ms |
   | decode-behind, double-buffered | ~35 ms | ~80 ms (+30) | ~98 ms (+40) | 35 ms |
   | decode-behind + first-frame fast path | ~37 ms | ~52 ms | ~70 ms | 37 ms; one ~2T gap per stream |

   - **Why L13 lost TTFA on the L4 (+50-60 ms p95).** The closed-loop client sends its next request when the
     previous one is reported done. L13 retires after the launch, so that request misses the step being launched
     and waits a full iteration.
   - **The decode-ahead loop removes this for free.** It retires *before* the pre-launch prefill pass, while decode k
     runs on the GPU, and can afford a short late wait there.
   - **First frame first (L13d)** puts the new rows' first frame at the head of the decode, so it reaches the client
     a few ms after its AR step instead of after the whole gang.
4. **Recommendation, in order:**
   - L13b (decode-ahead) on top of L13;
   - then L13c (prefill-behind, plus AR finish by event);
   - L13d is an opt-in TTFA knob.
   - The double-buffered decode-behind loop (section 6) is designed here as asked, but **not recommended**: no
     throughput over L13b + L13c, one more iteration of PCM latency, and three staging rings to get right.
5. **Against ping-pong (L26).**
   - Decode-ahead first. It is also the ping-pong plan's own Stage 1 (the decode split).
   - It keeps one full-width step, so ping-pong's extra fixed GPU cost per cycle (~3.8 ms) and its half-width
     padding do not apply.
   - CLI and C1 audio stay bit-identical.
   - The two compose but do not add up: both fill the same idle time. Ping-pong only wins if Stage 0 measures an
     exposed `e` above ~4 ms that prefill-behind cannot cover. Details in section 10.
6. **Cost.**
   - No new threads.
   - VRAM: ~0 for L13b/c; a few MB of small decoder graphs for L13d; ~50-100 MB for decode-behind.
   - Bit identity: C1 and CLI `--batch 32` identical. A new CLI streaming burst (`--batch 32 --stream`) is needed,
     because the offline CLI never runs the streaming decode.

---

## 1. What is measured, and what the code does today

### 1.1 Inputs

| input | value | source |
|---|---|---|
| host per iteration, all 11 flags, 2.6 GHz host, ~C800-900 | ~28 ms | box, `[SERVE] host ... ms per iteration` |
| decode-gang sync wait per iteration | ~11 ms | box, per-site table (`pocket_decode_audio_batch`) |
| AR one-sync wait per iteration | ~11 ms | plateau F5 / L5 (10.8-11.1 ms at C640) |
| other syncs | ~1-2 ms | 9.4-9.7 syncs per iteration in total with the flags (3d) |
| GPU busy (`dmon sm`) | ~65 % | box |
| serial loop identity | T = host + device wait + blocked | `[SERVE]` profile |
| admission | ~8 per iteration at ~900 rows, est. 1.4-2.2 ms each | host-profile note, section 4 |
| L13 on the L4 | +1 % aps, **TTFA p95 +50-60 ms** | plateau 3g |

### 1.2 Where the device waits are (code read)

- **AR, one sync.** `pocket_onesync_step` ends in one `mynah_backend_sync` (`cudaStreamSynchronize`). L13 split it
  into a QUEUE and a FINISH phase.
- **Gang decode, default path, also one stream sync.** In `pocket_decode_audio_batch`, for one frame per row
  (`audio_emit_frames` = 1), the path is:
  1. gang upsample (queued; event-guarded pinned meta);
  2. Mimi tile (queued; event-guarded tile meta);
  3. decoder batch graph (queued; L11 table patch, which waits on `entry->done` of the **previous** launch of that
     graph);
  4. `mynah_backend_gather_rows_d2h` (queued; one backend-wide pinned `pcm_host`, plus its `pcm_meta_event`);
  5. **one stream sync**;
  6. then the host post: PCM placement (L20), the finite scan, `pocket_decode_frame_finish`.

  The extra sync after the codec transformer (`engine_pocket.c` ~12810, quoted in the ping-pong note) is only on the
  opt-in `MYNAH_CUDA_CODEC_BATCH` path. The default Mimi-tile path has none.
- **Batched prefill, device-owned rows: no sync.** `pocket_prepare_slice_batch` → `pocket_cuda_prefill_tile` queues
  one tile over the whole text and marks the rows prepared at once. Only the non-device-owned fallback (a per-token
  `pocket_cuda_backbone_step_batch`) syncs.
- **Shared staging, event-guarded.** The tile workspace (`st->tile.meta_host` + `meta_event`) is shared by the Mimi
  tile and the prefill tile.
- **Safety net.** `pocket_cuda_ahead_discard` runs in `decode_audio_batch` and `prepare_slice_batch` (L13). The
  `step_launch` contract in `tts_engine.h` allows only `ctx_new*` / `ctx_attach` / `ctx_free` while a step is queued.

### 1.3 The cost model (to be replaced by Stage 0)

- **The serial identity.** T_off = H + W.
  - At ~C896: 28 + (11 + 11 + 2) ≈ **52 ms**.
  - GPU busy 65 % gives G ≈ 0.65 × 52 ≈ **34 ms** of GPU per iteration.
  - So ~10 ms of GPU work already runs under host time: kernels execute while the host is still queuing the decode,
    and the prefill tile runs under the AR launch.
- **Split, model only.**
  - G_d (decode) ≈ 16 ms.
  - G_a (AR + the prefill tile queued before it) ≈ 18 ms.
  - Host chain pieces at ~900 rows, from the host-profile estimates:

    | piece | est. |
    |---|---|
    | finish/commit | ~1 ms |
    | emit + gang selection | ~0.8 ms |
    | decode submit | ~2.5 ms (first kernel after ~0.5 ms) |
    | prefill pass (host) | ~1 ms |
    | select + AR launch staging | ~2.4 ms |
    | decode post | ~1.9 ms |
    | delivery (L19) | ~1 ms |
    | retire + cancellation | ~1.4 ms |
    | **admission** | **~14 ms** |
    | fixed / misses | ~2 ms |
- **~C768.** The per-row and per-admission terms scale with live rows (×0.86): H ≈ 24, G ≈ 29, T_off ≈ 45.
- **Calibration caveat.** The ping-pong note fits G(W) ≈ 3.8 + 0.054·W ms from `period = C·0.08/aps`. That gives a
  larger G (52 ms "at C896") because live rows per step are fewer than C (on the 10-05 logs, live ≈ aps·T/0.08 ≈
  0.8·C).
  - The two calibrations disagree on the absolute G, not on the ranking below.
  - With a larger G every pipelined variant is even more GPU-bound, and the differences between them shrink.
  - Stage 0 times AR and decode with events, and settles it.
- **Gate.** RTF p95 ≤ 0.88 needs a p95 iteration under ~70 ms. With p95/mean ~1.3, that is a mean iteration under
  ~54 ms.

---

## 2. Four loops on one stream

All on the one backend stream, `cuda_backend_state::stream`. "k" is the AR step. Iteration j finishes step k.

```
flag off      GPU: |pre AR k|            |dec k|                                   |pre AR k+1| ...
              host:  adm pre launch  wait  fin emit dsub wait post dlv ret         adm ...

L13 (coded)   GPU: |dec k|            |pre AR k+1|                 |dec k+1| ...
              host: fin emit dsub wait post dlv pre launch ret adm wait  fin ...

L13b          GPU: |dec k|pre AR k+1|                |dec k+1|pre AR k+2| ...
decode-ahead  host: fin emit dsub ret pre launch adm..(collect k: post dlv)..adm wait  fin ...
              (GPU idle only during e = fin + emit + decode-submit head, ~2-3 ms)

decode-behind GPU: |dec k-1|AR k+1|dec k|AR k+2|dec k+1| ...
(section 6)   host: fin emit ret pre launch dsub(k) wait(k-1) post dlv adm ... wait  fin ...
              (PCM of k collected in iteration j+1)
```

**Model iteration time T.**

| loop | rule | ~C768 | ~C896 | ~C1024 |
|---|---|---|---|---|
| off | H + W | 45 | 52 | 59 |
| L13 | off − min(retire + admission + cancellation, W_ar) | ~36.5 | ~42 | ~47 |
| L13b | max(H, G + e) | ~31 | ~36 | ~41.5 |
| L13b + L13c | max(H, G + max(0, e − G_prefill_behind)) + ~0.5 | ~30 | ~35 | ~40 |
| decode-behind | max(H, G) + ~1 | ~30 | ~35 | ~40 |

- Throughput relative to off at ~C896: L13 +23 %, L13b +44 %, L13b + L13c +49 %.
- RTF mean ≈ T/80: ~C1024 goes from 0.74 (off, fails the p95 gate) to ~0.5 (L13b + L13c).
- **Host-bound fallback.** When H > G (a 2.1 GHz host: H ≈ 54), L13b gives T ≈ H. The host never waits, because
  the GPU is always done first.
  - Decode-behind gives the same.
  - Ping-pong gives H plus a second set of per-group fixed host costs.
  - Only host diet helps there (host-profile A1-A8).

---

## 3. L13b: decode-ahead, collected in the same iteration (the recommended design)

### 3.1 Loop order (iteration j)

**On entry.**
- AR k is queued: launched in iteration j−1, behind decode k−1.
- Decode k−1 has already been collected and delivered.
- Rows whose last frame was k−1 are waiting to retire.

```
A. wait AR k          step_batch FINISH (commit, EOS logits, flow finite). Stream sync without L13c, event with it
B. emit k             EOS, budget, re-prepare; a row ending at k goes active=0, decoding=1 (its frame k is in the gang)
C. decode submit k    gang selection exactly as today (stream_gang policy), engine decode_submit, event dec_k recorded
                      (L13d: a small first-frame gang is submitted first, see 5.3)
D. late retire        rows with !active && !preparing && !starting && !decoding && !ahead: the rows whose final PCM
                      went out in j-1, the cancelled rows that have ridden their last step, the failed rows.
                      on_done runs here. Optional late wait (MYNAH_CUDA_FAST_FIRST_CHUNK_WAIT_US) if anything retired
E. prefill pass       sliced/batched prefill + late admission, as today (queued behind decode k, no sync)
F. select + launch    selection over the REAL arrangement (D already retired), step_launch(AR k+1)
G. admission, async collect, cancellation (every N)
                      between admission units: poll dec_k (cudaEventQuery). When ready, run H at once
H. decode collect k   wait dec_k if not yet (profiled sync site "pocket_decode_collect"), post (finite scan,
                      PCM placement, frame finish), deliver (slot_deliver → sink, L19 hand-off); decoding=0
I. L13c only          queue the batched prefill of the rows admitted in G right behind AR k+1 (5.2)
J. loop bookkeeping   requeue/prep_seq, profile; then back to A (the host waits for AR k+1 there)
```

**Hidden under the GPU.**
- Under decode k: D, E, F (retire, prefill, launch staging).
- Under AR k+1: G, H (admission, cancellation, decode post, delivery).
- Exposed: `e` = A + B + the head of C, until the first decode kernel is queued.
  - The head is `decode_admit` per row plus the O(n²) duplicate check. A3 in the host-profile note makes the check
    O(n).
  - Est. 2-3 ms at ~900 rows.
- With L13c, the prefill tiles queued at I run on the GPU during the next A + B + C. That covers `e` whenever at
  least a few rows were admitted.

**Why decode k goes before AR k+1 and not after it** (the decode-behind order):
- Decode k only depends on emit k, so it can be queued at C.
- Placed first, its PCM is ready ~G_d after the AR finished, inside the same iteration. That keeps the PCM latency
  of flag-off.
- Placed behind AR k+1, it would wait an extra G_a, and its collection would move into the next iteration (section
  6).

### 3.2 State and invariants (driver, `src/inference.c`)

- **New slot fields.**
  - `int decoding`: the slot has a frame in the gang that is in flight.
  - `size_t dec_pos`: its index in the pending gang.
  - Same pattern as L13's `ahead` / `ahead_pos`: retire's swap-remove moves slots, and collect re-maps them by
    scanning for `decoding`.
- **Pending gang record**, at most one in flight. A `dec_pending` struct holds:
  - `count`;
  - per-position `first`, `want`, `ctx`;
  - the emit flags needed after delivery (`eos`, `reprepare`).

  `step_live` keeps its post-emit state transitions. They depend only on emit, not on delivery, so they still run in
  B.
- **Retire predicate** gains `!decoding`. A row never leaves while its PCM is in flight. That is the same rule as the
  decoder lane's "never free state the lane is decoding". The context is freed only after collect.
- **Frame cursors.**
  - `streamed_frames` advances at submit, as the lane does today (`slot_emit` vs `slot_deliver`).
  - `streamed_samples` advances at delivery.
  - The ramp (`slot_quantum`) therefore never hands out a range twice.
- **Cancellation.**
  - A row cancelled in G with a frame in decode k still goes through collect. Its PCM is dropped, not delivered.
  - If it is also in AR k+1, it rides along exactly as in L13: it is marked failed at the next FINISH, and retired
    at the next D.
- **Failure at collect.** A non-finite PCM or a decode error for row r makes `pocket_decode_batch_drop` mark it
  broken, and `slot_fail` follows. If r is also in AR k+1, it rides along as a failed row and retires at the next D.
- **No retire simulation is needed.** In L13 retire ran after the launch, so `step_ahead_launch` simulated the
  swap-remove. Here D runs before F, so F selects over the real arrangement, which is the one flag-off would select
  over.
  - Rows ending at k are excluded from F because they are not active. They retire at D of iteration j+1.
  - In a burst (no admissions), the arrangement at D(j+1) is the one flag-off's retire produced at the end of
    iteration k. Section 7 relies on this.
- **Delivery order.** One gang in flight, collected before the next submit, so per-stream PCM order is trivially
  preserved. L19 pins a stream to one helper.

### 3.3 Engine hooks (`src/tts_engine.h`, appended)

```c
/* ---- decode split (APPENDED; MYNAH_CUDA_DECODE_OVERLAP) -------------------
 * OPTIONAL, both or neither. decode_submit queues exactly what
 * decode_audio_batch would compute for these ranges and returns without
 * waiting; decode_collect returns the same out_samples / out_count / failed
 * that decode_audio_batch would have returned. One gang in flight per scratch.
 * Between the two, the driver may call step_launch, prepare_slice_batch,
 * ctx_new*, ctx_attach, and ctx_free on contexts NOT in the gang; nothing else
 * on this scratch. collect with wait == 0 returns 1 when not ready, with
 * nothing changed. A submit that cannot queue (any per-row CPU fallback, a
 * multi-frame range, a host mirror) completes synchronously inside the call;
 * the following collect then returns at once. Bit-identical per context to
 * decode_audio (the decode_audio_batch contract). */
int (*decode_submit)(mynah_engine_ctx *const *ctxs, size_t count,
                     const size_t *first_frame, const size_t *frame_count,
                     int *failed, mynah_engine_scratch *scratch,
                     char *error, size_t error_capacity);
int (*decode_collect)(mynah_engine_ctx *const *ctxs, size_t count, int wait,
                      float **out_samples, size_t *out_count, int *failed,
                      mynah_engine_scratch *scratch,
                      char *error, size_t error_capacity);
```

The `step_launch` contract text gains: "decode_submit / decode_collect and prepare_slice_batch on device-owned rows
are allowed while a step is queued (L13b / L13c) and must not discard it."

### 3.4 Engine and backend changes (`src/engine_pocket.c`, `gpu/cuda/backend_cuda.cu`)

- **Split `pocket_decode_audio_batch`** (CUDA async branch, `longest == 1`, every row on the resident path) at the
  existing stream sync.
  - **Submit:** admit, gang upsample, prepare, Mimi tile, transform-finish bookkeeping, decoder submit (batch or
    per-row resident), gather D2H (or per-row `pocket_cuda_decoder_collect` for a 1-row gang), then **record the
    gang event**.
  - **Collect:** event wait or query, then everything after today's sync, unchanged, incl. `decoded_frames +=` at
    the end.
  - **Synchronous inside submit:** any other case (multi-frame ranges, CPU decoder fallback, codec-batch opt-in,
    debug dumps). Correct, just not overlapped. Counted as `sync submits`.
- **The gang lives in the scratch:** `scratch->dec_inflight` (ctxs, ranges, out pointers, flags, event).
- **`pocket_cuda_ahead_discard`.**
  - It stays in submit. At C no AR step is queued, so it never fires.
  - It is **removed from collect**. The AR step k+1 queued at F reads none of the decode's buffers:
    - AR reads the pinned latent-in that was copied at its launch;
    - decode writes codec rings, decoder I/O and `pcm_*`;
    - collect touches only host-side PCM and codec bookkeeping.
  - Audit item: confirm that no decode path touches `scratch->cuda_x / cuda_norm / cuda_proj / flow` buffers. Only
    the opt-in codec-batch path uses the scratch at all, and it uses its own `cuda_codec_*` arrays.
- **Backend event API.**
  - `mynah_backend_event_record(backend, slot)` / `_query` / `_wait` on a small pool of reusable events. Today
    `fence_record` creates and destroys an event per call.
  - `_wait` is a **profiled sync site**, so "device wait" stays comparable with older runs.
- **No buffer needs doubling for L13b.**
  - Only one decode is in flight, and decode k+1 is submitted only after decode k was collected.
  - The decoder table patch's `entry->done` wait, `pcm_meta_event` and `up_meta_event` all refer to work that is
    complete when the next submit runs.
  - The prefill tile at E waits on `tile.meta_event`, recorded by decode k's Mimi tile. That copy runs in the first
    ~0.5 ms of decode k, which has already started, so the wait is ~0.

---

## 4. L13c: prefill-behind, and AR finish by event

**What it does.**
- Rows admitted in G (after the launch) are prefilled at I, queued right behind AR k+1.
  - `pocket_prepare_slice_batch` on device-owned rows: one tile, no sync. The rows are marked prepared, so they are
    selected at F of iteration j+1, the same step they would have joined anyway.
- Their GPU time then fills the next iteration's exposed `e`.
- The prefill moves from the chain (E) into the GPU's shadow. For TTFA nothing changes: the rows join AR k+2 in both
  cases.

**What it needs.**
- **AR FINISH waits on an event** recorded after the AR's last D2H, not on a stream sync. Otherwise A would wait for
  the prefill queued behind AR k too.
  - This is the "fence-based finish" the ping-pong note also needs. It is byte-identical when nothing is queued
    behind.
- **`prepare_slice_batch` must not discard** a queued step when every row is device-owned and not queued. The tile
  touches only the new rows' KV and the tile workspace, never the AR scratch. Any other row set keeps the safety net.
- **Only rows that need no host-side fallback.** A non-device-owned row, or a scalar `prepare_slice`, waits for E as
  today.

**Identity.** The prefill batch composition changes, which rows go in which tile. The batched-prefill contract says
the result must not depend on gang width or row order, so the expectation is id per row. Treat it as inv under
concurrency. A burst is unaffected: everything is prefilled before any step is queued.

---

## 5. Latency: TTFA and the gap between chunks

### 5.1 What sets TTFA

For a request that needs one prefill tile (the CUDA device-owned path does the whole text in one tile):

TTFA ≈ (wait from arrival to the next **pre-launch pass**) + (pass → first PCM delivered).

The pass is the last point where a new row can still join the step about to be launched (the late admission in
`prefill_pass`).

| loop | where the pass is | pass → delivery | arrival → pass, Poisson | arrival → pass, closed loop |
|---|---|---|---|---|
| off | after the 14 ms admission, at ~0.28 T | launch + G_a + fin + emit + G_d + post ≈ **32 ms** | T/2 ≈ 26 | ~2-5 ms: retire at the end of the iteration, then the request lands in the next top admission |
| L13 | before the launch, after delivery (~0.5 T) | (T − pass) + fin + emit + G_d + post ≈ 42 | T/2 ≈ 21 | **~37 ms**: retire after the launch, so the re-request misses that pass and waits a whole iteration |
| L13b | at E, ~0.15-0.25 T, inside decode k's shadow | (T − pass) + e + G_d + post ≈ 48 | T/2 ≈ 18 | ~0-3 ms with late retire (D) + a ≤ 3 ms late wait; ~10 ms without the wait |
| decode-behind | before the launch, ~0.13 T | ~2.3 T ≈ 81 (AR behind decode k−1; decode behind AR k+2; collected the iteration after) | T/2 ≈ 17.5 | ~0-3 ms |

**Model TTFA, mean, n_prefill = 1, client round trip ~2 ms, in ms:**

| arm | ~C768 closed | ~C768 Poisson | ~C896 closed | ~C896 Poisson |
|---|---|---|---|---|
| all 11 flags | ~43 | ~50 | ~50 | ~58 |
| + L13 | ~69 (+26) | ~54 (+4) | ~80 (+30) | ~63 (+5) |
| + L13 + L13b | ~43 (0) | ~58 (+8) | ~50 (0) | ~67 (+9) |
| + L13c | ~41 | ~56 | ~48 | ~65 |
| + L13d (first frame first) | **~30** | **~45** | **~35** | **~52** |
| decode-behind (double-buffered) | ~69 | ~84 | ~80 | ~98 |
| decode-behind + fast frames (N=2) | ~45 | ~60 | ~52 | ~70 |

**Reading the table.**
- **p95.** On the L4, L13's measured penalty at p95 (+50-60 ms) was ~1.5× this mean model's one-iteration
  estimate. Expect p95 deltas ≈ 1.5× the mean deltas. The ranking is what matters.
- **The closed loop (the ladder) and Poisson traffic disagree on purpose.**
  - Closed-loop TTFA is dominated by the *phase*: where `on_done` lands relative to the next pre-launch pass.
  - Poisson TTFA is dominated by T/2 + pass→delivery.
  - **Both are reported in the box plan.**
- **Why L13b keeps flag-off's closed-loop TTFA.**
  - Retire (and `on_done`) moves to D, right before the pre-launch pass. The re-request (~1-3 ms on loopback)
    either lands before the pass or is caught by the late wait.
  - The late wait is free in L13b: it runs while decode k occupies the GPU. The chain D + wait + E + F (~1.4 + ≤ 3
    + 1 + 2.4 ms) fits well inside G_d ≈ 16 ms.
  - In L13 alone the same reordering would cost GPU idle time (decode is synced there), which is why L13 put retire
    after the launch.
- **Why Poisson TTFA is +9 ms in L13b.** A new row's first AR step is queued behind decode k (G_d ≈ 16 ms). Its
  first frame is then decoded in a full-width gang. Flag-off pays neither, but waits longer for the pass (T/2 =
  26 vs 18).
- **"Admissions ride the queued step when a slot is free at launch time."**
  - A launched step cannot take extra rows: its tables, staging and graph are fixed at the launch.
  - So "riding" means making the pre-launch pass the place where those requests land. That is late retire + late
    wait (closed loop) and an early pass (Poisson).
  - For L13 alone the equivalent fix is to retire before the pre-launch pass. It costs ~1.4 ms plus the wait of GPU
    idle per iteration; A/B it as `MYNAH_CUDA_STEP_OVERLAP=2` only if L13b slips.

### 5.2 Mitigations, cheapest first

1. **Late retire before the pass (L13b default).** Free in L13b. Needed for closed-loop TTFA.
2. **Late wait.** `MYNAH_CUDA_FAST_FIRST_CHUNK_WAIT_US`, existing; default 0; A/B 2000 / 3000.
   - Only after a retirement.
   - L13b caps it so that D + wait + E + F stays under the decode's EWMA GPU time. Over the cap the wait is
     skipped, and that is counted.
3. **Poll-collect between admission units.** The decode k event is queried between the ~1.8 ms context builds, so
   delivery happens within ~2 ms of the decode finishing, instead of after the whole ~14 ms admission pass.
4. **Cap chain admissions.** Late admissions at E build their context in the chain (~1.8 ms each).
   - Cap them with `MYNAH_CUDA_PIPE_CHAIN_ADMITS`, default 4. The rest are admitted at G, in the shadow, and still
     join the next step.
   - With host-profile A1 (cheap admission) the cap stops mattering.
5. **L13d, first frame first.** Rows whose `streamed_frames == 0`, about one per admission (~8 per iteration):
   - their frame k goes into a small gang submitted **before** the main gang at C, with its own event;
   - it is collected right after C (it runs during the main gang's host submission) and delivered before D;
   - first PCM moves from ~e + G_d + post (~20 ms after the AR) to ~e + g_fast + ε (~5-7 ms).
   - **Cost.** One extra small decode per iteration, g_fast est. 1-3 ms of GPU: launch-bound SEANet + Mimi tile +
     gather at width ~8. The main gang narrows by the same rows.
   - **Small widths change every iteration.** Run them eager below width 16, or pad to buckets 8/16/32 (L25), so
     they do not churn decoder graph captures.
   - **First gap.** Frame 1 goes through the main gang one iteration later, ~14 ms later in the iteration than the
     fast gang delivered frame 0. So the first gap is ~T + 14 ms once. The 80 ms frame already in the client's
     buffer covers it with room to spare.

### 5.3 The gap between chunks

- **Steady state.** Every live row gets one frame per iteration (`audio_emit_frames` = 1), delivered at the same
  phase of each iteration. The gap is T: 52 → ~36 ms at ~C896.
  - gap p95 tracks T p95. It improves with every pipelined variant.
  - stalls@250 can only fall.
- **A one-iteration PCM lag** (decode-behind) is a constant phase shift. It does not change the steady gap. It
  costs:
  - one more iteration of TTFA;
  - one more iteration before the stream end (the last PCM, then `on_done`);
  - a one-time 2T gap whenever a stream switches from a fast first-frame path to the lagged path. That is why
    decode-behind wants N=2 fast frames, so that the buffered lead (160 ms − T) absorbs it.
- **L13b's stream end.** The last PCM goes out at H of the iteration where it was emitted. `on_done` follows at D of
  the next one, ~15 ms later. Harmless; the HTTP terminator comes slightly later.

---

## 6. The asked-for variant: decode-behind, PCM one iteration later, double-buffered

Designed in full so it can be built if Stage 0 contradicts section 2 (for example, if `e` turns out large and
prefill-behind cannot cover it). Flag `MYNAH_CUDA_DECODE_OVERLAP=2`.

### 6.1 Loop order (iteration j)

**On entry.** The GPU queue holds [decode k−1 main] (ring slot r), started when AR k finished.

```
A. wait AR k by EVENT (decode k-1 keeps running), finish k
B. emit k
C. fast gang (MYNAH_CUDA_FAST_FRAMES=N, default 2): rows with streamed_frames < N, submitted now, ring slot f
D. late retire (rows whose final PCM went out in j-1), late wait
E. prefill pass + capped late admission
F. select + launch AR k+1           (behind decode k-1 and the fast gang)
G. submit decode k main, ring slot r' = r^1     (behind AR k+1; graph-table parity r')
H. collect decode k-1 (event r), fast gang (event f); post + deliver; rows whose final frame was k-1 retire at D(j+1)
I. admission, cancellation, async collect; [prefill-behind]
```

### 6.2 What has to be doubled, or ringed

Backend-wide today, written by the host while an earlier decode may still be queued:

| state | file | ring |
|---|---|---|
| gang PCM target `pcm_dev` / `pcm_host` (gather + D2H) | `backend_cuda.cu` `cuda_codec_gang_workspace` | 3: main ×2 + fast; **mandatory**, or decode k's D2H overwrites the PCM of k−1 before H reads it |
| gather pointer meta `pcm_meta_dev` / `pcm_meta_host` + `pcm_meta_event` | same | 3. Without it, G blocks until decode k−1 reaches its gather |
| upsample meta `up_meta_*` + event | same | 3 |
| tile meta `tile.meta_host` + `meta_event` (Mimi tile and prefill tile) | `cuda_tile_workspace` | 3. The prefill tile at E would otherwise wait for decode k−1's tile meta |
| decoder batch graph pinned pointer tables (L11 patch target), `entry->done` | `cuda_decoder_batch_graph_entry` | parity 2. Preferred: two pinned table sets per entry, switching the graph's memcpy-node source with `cudaGraphExecMemcpyNodeSetParams1D` (no re-instantiate). Fallback: a lane/parity in the entry key (2 execs per hot width) |
| per-context device buffers (codec ring, decoder in/out, `cuda_codec_up`) | engine | none: stream order serialises decode k−1 → decode k per context |
| per-context `lent_pcm` (L20) | engine | none: written only at collect, delivered at once |
| AR scratch (latent-in, noise, EOS/flow D2H) | engine | none: one AR in flight |

**Events.**
- `ar_done[k]`, waited at A.
- `dec_done[slot]`, waited at H.
- One per staging-ring slot, waited only when that slot is reused (two submissions later, so always complete in
  steady state).
- L6 park fences, as today.

### 6.3 Engine bookkeeping that moves

- The codec position (`mynah_seanet_state_advance`) and `decoded_frames` advance at **submit**.
  - Decode k is prepared before decode k−1 is collected, and `pocket_decode_frame_prepare` checks that the SEANet
    position equals the transformer offset.
- The finite scan and `broken` stay at collect.
  - A row whose frame k−1 fails has frame k already queued. Its PCM is dropped at the next collect and the row
    retires.

### 6.4 Its latency, and why it is not recommended

- PCM of step k arrives ~(G_a + G_d + host) after AR k, about 1.3-1.6 T later than L13b. That is +30-40 ms of TTFA at
  ~C896 (section 5 table).
- The fast path (N=2) recovers the TTFA but:
  - costs g_fast of GPU per iteration;
  - adds a 2T gap per stream at the switch;
  - needs its own ring slot.
- Its throughput over L13b + L13c is at most `e` minus the prefill-behind cover, ≈ 0-3 %.
- **Revisit only if** Stage 0 shows an exposed GPU idle above ~4 ms per iteration with L13b + L13c on.

---

## 7. Audio identity

| check | expectation | why / what it must show |
|---|---|---|
| server C1, `pocket_ladder` fixed seed: all flags + L13 vs + L13b (+ L13c, + L13d) | **id** | Width 1: no AR launch (L13 refuses below 2). The decode of a 1-row gang is still submitted async (per-row resident decoder + D2H + event): same kernels, same order. `[SERVE] decode overlap` must show async gangs > 0 |
| CLI `--synthesize --batch 32`, seeds 1000-1031 (offline) | **id** 32/32 | Offline rows have no callback and are decoded once at retire (`slot_finalize` → `decode_audio`). L13b changes only where retire sits, and D-before-F selects over the same arrangement as flag-off. Must show L13's `launched > 0`; async gangs = 0 is expected. **This gate does not exercise L13b** |
| **new: CLI streaming burst** `--synthesize --batch 32 --stream` (jobs with a callback writing to the WAV) | **id** 32/32 against flag-off | Everything admitted at once, no arrivals. The gangs have flag-off's members, ranges and order. Rows ending at k retire at D(j+1), giving the same arrangement flag-off's retire gave at the end of iteration j. With L13d the first frames form one 32-wide fast gang, which is exactly flag-off's first gang |
| same, with `MYNAH_CUDA_DECODE_CHECK=1` (debug: a full stream drain after every submit and launch) | byte-identical to the run without it | a race on a shared pinned buffer would show here |
| server C64 / C256 under concurrency | **inv** | composition follows timing (plateau 3b) |

- **The decode split is numerically a no-op.** The seam already requires `decode_audio_batch` to be bit-identical per
  context to `decode_audio`, and the CLI self-check verifies it.
- **What can differ under concurrency** is the AR composition (an admission joins one step later than flag-off) and
  the prefill grouping (L13c). Both are the existing inv class.

---

## 8. Interaction with the flags

| flag | with L13 + L13b (+ L13c) |
|---|---|
| ONE_SYNC (default on) | required: the launch is its queue half |
| **L13** `STEP_OVERLAP` | required. `DECODE_OVERLAP` is refused, with a start-up line, without it. L13's retire simulation stays for the case where D is skipped |
| **L6** deferred release | **required.** A `ctx_free` drain at D would wait for decode k, and at G for AR k+1, so every retire would serialise. With L6 the park is fenced and re-take waits on its fence (rare) |
| L7 cancel every N | unchanged. A cancellation noticed at G rides AR k+1 and drops its in-flight PCM |
| L8 writev | unchanged (writer threads) |
| L10 dup-check epoch | unchanged for the step. The decode-gang dup check is still O(n²) (host-profile A3): fix it, because it sits in `e` |
| **L11** decoder table patch | unchanged in L13b: decode k+1 is submitted after decode k completed, so the `entry->done` wait is ~0. Decode-behind needs the parity tables (6.2) |
| L12 survivor flow reuse | unchanged (the flow layout is computed at the launch, after emit) |
| **L19** deliver threads | delivery moves to H, under AR k+1, off the critical path. Its value on the scheduler thread shrinks, while its helpers still use CPU. **A/B L19 = 0 / 2 / 4 under L13b**, and on 4 vCPU prefer 1-2. The hand-off copies the PCM synchronously, so the gang pinned buffer is free when `slot_deliver` returns |
| **L20** PCM direct | unchanged: `lent_pcm` per context is filled at collect from the single pinned gang buffer and delivered at once. Under decode-behind the gang buffer is the ring slot of k−1; `lent_pcm` stays single |
| L21 lazy hidden | emit (B) settles the lazy rows before the launch (F), as in L13. The rare copy-back fallback D2H is a stream sync and would wait for decode k: correct, counted as a bubble |
| L22 KV table cache | unchanged (AR only) |
| L24 validate once | unchanged |
| async admission (`MYNAH_ASYNC_ADMIT`) | compatible (`ctx_attach` takes no scratch). Not recommended on 4 vCPU |
| KV growth | a refused launch (KV grow) gives a serial step that iteration. Its stream sync then waits for decode k: correct, slower. `MYNAH_CUDA_KV_VMM=1` + pre-grow (ping-pong note 4.6) keeps refusals rare |
| decoder lane, parity dump, `MYNAH_PREFILL_SLICE=0` | refused, as for L13 |

---

## 9. CPU budget, VRAM, risks

### 9.1 CPU threads

- **No new threads.** Events plus `cudaEventQuery` polling on the scheduler thread: ~10 queries per iteration at
  ~1-2 µs.
- **The scheduler thread** goes from ~54 % real work (the rest spin-waiting) to ~80-90 %.
  - `MYNAH_CUDA_SYNC=yield|blocking` is worth an A/B, as in the ping-pong note.
  - With the late wait and poll-collect, the remaining waits are short.
- **Per-audio-second work grows with throughput (+40-50 %)** on threads that already exist: the L19 helpers, the
  ~900 writer threads (host-profile A9) and the HTTP workers.
- **On a 4-vCPU host:**
  - pin the scheduler to one physical core;
  - L19 at 1-2;
  - the load client off the box.

  The 2026-10-04 run had the client at ~100 % of a vCPU on the same host.

### 9.2 VRAM

| item | L13b | L13c | L13d | decode-behind |
|---|---|---|---|---|
| events | a few | 1 | 1 | ~10 |
| PCM gather buffers | 0 | 0 | 0 (the fast gang reuses the slot after the main collect) | +2 × (rows × 1920 × 4 B) dev + pinned ≈ +32 MB at 1024 rows |
| staging rings | 0 | 0 | 0 | < 5 MB |
| decoder graphs | 0 | 0 | small-width buckets, a few MB | parity tables (MBs), or 2 execs per hot width (tens of MB) |
| **total** | **~0** | **~0** | **< 10 MB** | **~50-100 MB** |

### 9.3 Risks

1. **Hidden stream syncs while work is queued.**
   - Any `mynah_backend_sync` reached in D-I waits for decode k and/or AR k+1. The overlap of that iteration is
     silently gone.
   - Candidates: admission's device half (slot-pool take, voice copy, decoder warm-up), `ctx_free` without L6, KV
     grow, lazy-hidden fallback, the non-device-owned prefill, decoder graph forget/destroy.
   - Mitigation: a per-site counter "syncs while queued" (the driver sets a queued-depth flag that
     `mynah_backend_sync_at` reads). Fix any recurring site before judging speed.
2. **Collect placed too late** → TTFA. Poll-collect between admission units bounds it to ~one context build
   (~2 ms). The profile reports ready-on-poll vs waited, and the mean collect delay.
3. **A decode path that does touch AR scratch** (3.4 audit). It would corrupt the queued AR. The streaming-burst
   identity plus `DECODE_CHECK` catch it.
4. **Partial submissions.** The CPU fallback, multi-frame ranges and the codec-batch opt-in complete inside submit.
   That is correct, but they serialise. Counted (`sync submits`); expect ~0 in steady state.
5. **The late wait as a benchmark artefact.** It helps closed loops, not Poisson traffic. It is capped by the decode
   shadow, so it never costs GPU time. Default stays 0. Report both traffic shapes.
6. **L13d's small-width decoder graphs** churn without buckets. Eager below 16, or bucket.
7. **The open 1024-row crash** (plateau 3e item 5). Reproduce it with all flags before trusting C1024 numbers.
8. **Model error.** G may be 1.5× larger (section 1.3). That changes absolute T, not the ranking.

---

## 10. Against ping-pong (L26, `.work/pocket-l40s-1024-pingpong.md`)

| | L13 + L13b (+ L13c) | ping-pong (two groups) |
|---|---|---|
| what stays exposed (GPU idle) | `e` = finish + emit + decode-submit head, ~2-3 ms; covered by L13c's prefill whenever there are admissions | ~0 while H/2 ≤ G(W/2) |
| extra GPU per cycle | 0 (one full-width step and decode) | +~3.8 ms (second fixed cost) + half-width bucket padding |
| model T at ~C896 / ~C1024 | ~35 / ~40 | ~38 / ~44 (calibration A); the ping-pong note's own: ~66-70 at C1024 against L13b's 59 + e |
| host-bound host (2.1 GHz) | T ≈ H (no waits) | T ≈ H + a second set of per-group fixed host costs (~+8 ms) |
| TTFA | closed loop ≈ flag-off; Poisson +9 ms (−6 with L13d) | their estimate: flag-off +0-30 ms |
| identity | C1 and CLI burst id, plus a streaming burst id | CLI id only below `PINGPONG_MIN`. Above it, a "split identity" against two half bursts |
| VRAM | ~0 | +0.1-0.3 GB (second graph set) |
| code | small driver change on top of L13 + the decode split | groups, lanes, scratch ×2, admission balancing, half-width buckets |

**The ping-pong note's decision rule needs one correction.**
- It reads L13b as "emit + decode preparation + **launch preparation** exposed" (`e ≥ ~6 ms` → ping-pong).
- In decode-ahead order, launch preparation (staging, noise, KV tables, graph launch) runs while decode k is on the
  GPU, and so does the prefill pass.
- What stays exposed is finish + emit + the decode-submit head, and L13c covers even that with queued prefill work.

**The corrected rule.** Measure the actual GPU idle per iteration with L13b + L13c (event timestamps: AR end →
next kernel start).
- Ping-pong is worth building only if that idle **exceeds its own extra GPU cost** (~3.8 ms per cycle, ping-pong
  Stage 0a).

**Order.**
1. The decode split. It is shared: it is ping-pong's Stage 1 too.
2. L13b.
3. L13c.
4. Measure.
5. Ping-pong only if the rule says so.

**Do they compose?** Mechanically yes: each group could run decode-ahead internally. They hide the same idle time,
though. Once one of them makes the loop GPU-bound (G ≥ H), the other adds only its own cost. When the loop is
host-bound (H > G), neither helps and only host diet does. **Do not stack them.**

**Shared foundations** (build once, used by both):
- fence/event-based AR finish;
- the backend event pool, and event waits as profiled sync sites;
- the "syncs while queued" counter;
- `decode_submit` / `decode_collect`;
- the streaming-burst CLI identity gate;
- `MYNAH_CUDA_KV_VMM` with pre-grow.

---

## 11. Flags

| item | flag | default | audio | notes |
|---|---|---|---|---|
| L13b | `MYNAH_CUDA_DECODE_OVERLAP=1` | 0 | id at C1 / CLI / streaming burst; inv under concurrency | needs L13 + L6 + the engine split; refused with the decoder lane, parity dump or `PREFILL_SLICE=0` |
| L13c | `MYNAH_CUDA_PREFILL_BEHIND=1` | 0 | id per row expected; inv | needs L13; also switches the AR finish to an event |
| L13d | `MYNAH_CUDA_FIRST_FRAME_FIRST=1` | 0 | id (burst); inv | needs L13b; small-width bucket or eager decoder below 16 |
| chain admissions cap | `MYNAH_CUDA_PIPE_CHAIN_ADMITS=N` | 4 | id | late admissions built in the chain (E) |
| late wait (existing) | `MYNAH_CUDA_FAST_FIRST_CHUNK_WAIT_US` | 0 | id | A/B 2000 / 3000 with L13b; capped by the decode shadow |
| debug | `MYNAH_CUDA_DECODE_CHECK=1` | 0 | — | drain after every submit / launch; must be byte-identical |
| decode-behind (not planned) | `MYNAH_CUDA_DECODE_OVERLAP=2`, `MYNAH_CUDA_FAST_FRAMES=N` | — | — | section 6, only if Stage 0 asks for it |

**Profile line.**

```
[SERVE] decode overlap (MYNAH_CUDA_DECODE_OVERLAP): gangs N async / M sync-submit, collect ready-on-poll X% /
        waited Y% (mean wait W ms, mean submit→deliver D ms), late retire R, late waits hit H / skipped S,
        chain admits C, fast gangs F (mean width w), syncs while queued Q (by site)
```

---

## 12. Staged implementation plan

| stage | content | files | effort | gate |
|---|---|---|---|---|
| **S0** measurement first | event pool + event waits as profiled sync sites; "syncs while queued" counter; event-timed spans per iteration (AR, decode, the idle between AR end and the next kernel = exposed `e`), sampled every Nth iteration; the `[SERVE] decode overlap` skeleton; CLI `--batch N --stream`. Ideally together with the host-profile level 2 (`MYNAH_SERVE_PROFILE=2`) | `src/backend.{h,c}`, `gpu/cuda/backend_cuda.cu`, `src/inference.c`, `cli/main.c` | S-M, ~1 day | no behaviour change; CLI and C1 identical to today |
| **S1** engine split | `decode_submit` / `decode_collect` for Pocket CUDA (single-frame resident ranges async, everything else synchronous inside submit); driver calls them back to back when `DECODE_OVERLAP=1`, same loop order | `src/tts_engine.h`, `src/engine_pocket.c` | M, ~1.5 days | streaming burst + C1 identical; CPU build unaffected (hooks NULL) |
| **S2** L13b loop | the order in 3.1; `decoding` / `dec_pos`; late retire at D; poll-collect inside `admit_pass` (a callback every unit); cancel/fail at collect; chain admission cap; late-wait cap; profile line. `tests/test_driver.c`: fake engine with submit/collect (order, retire after final PCM, cancel while decoding, failure at collect, burst identity, C1 identity, sync-submit fallback) | `src/inference.c`, `tests/test_driver.c` | M, ~1.5 days | driver tests green; on the box, the identity set of section 7 |
| **S3** L13c | AR finish by event; prefill-behind at I with the relaxed safety net for device-owned rows | `src/engine_pocket.c`, `src/inference.c` | S-M, ~0.5-1 day | identity; exposed `e` → ~0 in the S0 spans |
| **S4** L13d | first-frame gang before the main gang; eager or bucketed small-width decoder | `src/inference.c`, `src/engine_pocket.c`, maybe `backend_cuda.cu` | S-M, ~1 day | TTFA A/B |
| S5 (conditional) | decode-behind (section 6) or ping-pong, by the rule of section 10 | — | M-L | — |

Host-profile A1 (cheap admission) and A3 (decode dup check O(n)) are independent and compound with this.
- A3 shrinks `e`.
- A1 matters for slow hosts, where the loop goes host-bound.

---

## 13. Box test plan

**Box and build.**
- One L40S on a host of ≥ 2.6 GHz.
- The server pinned to the GPU's NUMA node (`knee.sh` `PIN=`).
- Clients on other cores, `--client-procs 4`.
- `make cuda cuda-server CUDA_ARCH=sm_89 ROW_CAP=1024`.
- Every job time-boxed with a capped waiter; a stop also kills its `knee.sh`.

**Common environment.** The 11 flags (ALL), `MYNAH_CUDA_KV_VMM=1`, `MYNAH_SERVE_PROFILE=1` (2 if available), the
same explicit width-bucket list in every arm.

### 13.1 Identity (before any speed run)

**Arms.** ALL + L13 (reference) vs ALL + L13 + L13b; then + L13c; then + L13d.

1. **CLI offline** `--synthesize --batch 32`, seeds 1000-1031.
   - Expect 32/32. `launched > 0`; async gangs = 0.
2. **CLI streaming burst** `--batch 32 --stream`.
   - Expect 32/32 against flag-off and against L13.
   - The arm must show async gangs ≈ steps, sync submits = 0, discard lines = 0.
3. **The same with `MYNAH_CUDA_DECODE_CHECK=1`.** Byte-identical to 2.
4. **Server C1, `pocket_ladder`, fixed seed, `--save-audio`.**
   - Expect all identical.
   - Async gangs > 0.
5. **Server C64.** Same length per request; the self-check passes.

### 13.2 Speed

**Levels and arms.** 2-minute levels, C768 → C896 → C1024 (then C1152 if C1024 passes). One server per arm. Two
repetitions, arm order alternated (ABCD DCBA).

| arm | env |
|---|---|
| A | ALL |
| B | ALL + L13 |
| C | ALL + L13 + L13b (+ late wait 3000 µs) |
| D | C + L13c |
| E | D + L13d (C896 and C1024 only) |
| C0 | C with the late wait at 0 (C896 only) |

**Per level, report:**
- audio-s/s, RTF p95, TTFA p50 / p95 / p99, gap p95 and max-gap p95, stalls@250, failures;
- `dmon sm`, W;
- host per iteration, device wait split into stream and event, syncs per iteration by site, **syncs while queued**;
- the S0 spans: AR ms, decode ms, exposed `e` ms;
- the `decode overlap` line;
- `launched` / `refused`;
- `/debug/first-chunk`: the `first_iters` histogram 0/1/2/3+ and the arrival offsets;
- VRAM at ready and at the end;
- server CPU per thread.

**Poisson arm.** `pocket_ladder --mode poisson` at the request rate arm A reached at C896 (read off its summary),
arms A, B, D, E. Poisson TTFA is the production-relevant number; the closed loop overstates the phase effect.

### 13.3 Pass criteria (L13b + L13c, arm D)

- C1024: RTF p95 ≤ 0.88 in both repetitions, 0 stalls, 0 failures.
- `dmon sm` ≥ 88 %, exposed `e` ≤ 1 ms per iteration.
- TTFA p95, closed loop: ≤ A + 15 ms, and ≤ B − 25 ms at C768 and C896.
- TTFA p95, Poisson: ≤ A + 15 ms.
- gap p95 ≤ A at every level.
- Syncs while queued ≈ 0 per iteration; no discard line; sync submits ≈ 0.
- VRAM delta ≤ 0.1 GB.

**Decision.**
- If D passes, ping-pong becomes the slow-host option at most.
- If the exposed idle stays above ~4 ms, apply the rule in section 10.

### 13.4 Then

1. **10-minute soak** at C1024, arm D. Watch the 1024-row crash, VRAM drift and the discard line.
2. **4-vCPU emulation.** Server pinned to 2 cores + SMT, clients elsewhere.
   - A vs D at C640 / C768;
   - L19 = 1 / 2 / 4;
   - `MYNAH_CUDA_SYNC` default vs `blocking`.
3. **L4 knee** C256 / C288 / C320, arms A, B, D: the 3c close-out rule.
   - On the GPU-bound L4 expect a small gain, and TTFA back to A's level, where L13 alone cost +50-60 ms.
