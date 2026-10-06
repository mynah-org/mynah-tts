# Pocket 24L on one L40S: the throughput plateau is the serial host loop

Board: `PLAN.md` E15-32. Branch `pocket-l40s-plateau`. Opened 2026-10-05.

On an L40S the CUDA server stops near 720-880 audio-s/s, whatever the concurrency, with the GPU idle about
40 % of the time. This note records what was measured, what was ruled out, and the work items. Each item is
an A/B flag that defaults to today's behaviour until it is measured. Tick an item when it is measured, and
write WIN or KO with the numbers.

## 1. Measurements

All 24L English, current CUDA defaults (int8 KV etc.), `tools/pocket_ladder.py` closed loop, v2 corpus, four
voices, seed 1234, 2-minute levels, 0 failures and 0 stalls@250 everywhere unless noted.

**2026-10-04: one L40S on a 4-vCPU host (2 cores + SMT), client on the same host:**

| build | C | audio-s/s | stream RTF p95 | TTFA p95 | GPU W / `dmon sm` |
|---|---|---|---|---|---|
| main (384 rows) | 384 | 740 | 0.488 | 87 ms | ~241 / 62 % |
| rows 768 | 512 / 640 / 768 | 715 / 718 / 726 | 0.667 / 0.826 / 0.970 | 118 / 147 / 175 ms | 238 / 60 % |

**2026-10-05: one L40S (Vast.ai, `sm_89`) on a host with a ~30-CPU container quota, client on the same host:**

| build | C | client procs | audio-s/s | stream RTF p95 | TTFA p95 | GPU W / `dmon sm` |
|---|---|---|---|---|---|---|
| rows 768 | 640 (fresh server) | 1 | 827 | 0.794 | 137 ms | 261 / 63 % |
| rows 768 | 640 (fresh server) | 4 | 808 | 0.799 | 133 ms | 261 / 62 % |
| rows 768 | 384 / 512 / 640 / 768 (one server, ladder) | 4 | 723 / 834 / **877** / 813 | 0.506 / 0.648 / **0.680** / 0.950 | 88 / 106 / 115 / 156 ms | 239-277 / 58-66 % |

## 2. Findings

- **F1 — `dmon sm` measures time, not occupancy.** It is the share of time at least one kernel runs, so 60 %
  means the GPU is completely idle 40 % of the time. The kernels do not "underfill" the GPU; nothing is queued.
- **F2 — the load generator is not the limit (CONFIRMED).**
  - On an M1, against `tools/gpu/stub_stream_server.py` (paced at RTF 0.5), one client process carries C512
    (12,800 chunks/s) at RTF 0.500 using 22 % of a core.
  - On the L40S, 1 and 4 client processes give the same result (827 vs 808).
- **F3 — host CPU was part of the 4-vCPU ceiling, not the main one.**
  - With ~30 CPUs, C512-640 gain 17-22 % throughput, and C640 RTF p95 drops from 0.83 to 0.68.
  - C384 does not move (740 vs 723).
  - The plateau remains at ~830-880 audio-s/s with the GPU ~63 % busy.
- **F4 — the serving loop is strictly serial, and the host half is the larger one.** From `MYNAH_SERVE_PROFILE`
  at C640:
  - device wait (time inside `mynah_backend_sync`) 44.7 % of the loop;
  - host 55.1 %, 31.7 ms per iteration, while the GPU has nothing queued;
  - iteration ~57 ms, step at width 640 ~47 ms.
- **F5 — where the syncs are (per-site profile, C640, first run).**
  - *Caveat:* in this first run the per-site table counted from process start, while the loop shares count from
    loop start. Start-up work was therefore charged to the table, and it inflated "device wait" by up to ~10 s
    of the 171 s. Since fixed: the counters reset when the serving loop starts. Re-measure.
  - With that in mind:

| site | per iteration (as printed) | mean wait | reading |
|---|---|---|---|
| `pocket_onesync_step` | 0.99 | 10.8 ms | the AR step: real, unavoidable without pipelining |
| `pocket_decode_audio_batch` | 0.97 | 11.7 ms | the gang decode: real, once per step (one frame per step, `audio_emit_frames=1`) |
| `pocket_cuda_decoder_step` (single context) | 15.9 | 0.21 ms | **almost certainly start-up**: 47k calls ≈ 10 s, i.e. slot-pool / decoder warm-up during the 28 s to ready. The streaming gang never reaches it; its fallbacks use submit without a sync. Confirm on the re-run |
| `pocket_cuda_drain_before_release` | 5.5 | 0.004 ms | a drain on every context free / park: no wait, but avoidable (L18) |
| `pocket_cuda_flow_step_batch` | 0.88 | 0.41 ms | not prefill. `pocket_emit_batch` reruns the flow head on the survivors whenever any row in the step is terminal, because one-sync flow needs `gathered == count`. With ~5.5 completions per iteration that is almost every step (L12) |

  - **The decoder graph is re-recorded on essentially every step.** Gang membership changes every iteration
    (retire, admit, swap-remove), so the exact match fails and the shape-reuse path re-records the whole SEANet op
    sequence: BeginCapture, full impl, EndCapture, GraphDestroy. It does this only to rewrite pinned tables (L11).
- **F6 — the model.** One iteration is about 22-26 ms of GPU work plus about 32 ms of host work, in series.
  - On an L4 the GPU part is ~2.5-3x longer, so the same host time is only 10-15 % of the loop. That is why the
    L4 runs 85-95 % busy and the L40S does not.
  - Upper bound with perfect overlap: max(GPU, host) per iteration, ~1.8x.
  - Host-diet items L7-L12 and L18-L25 are estimated at 7-13 ms of the ~32 ms, so +20-35 % at C640 and GPU busy
    ~62 → ~75 %. Overlap (L13) hides most of the rest.

## 3. Work items

Status: `[ ]` open · `[~]` coded, not measured on a GPU · `[x]` measured (WIN / KO) · `[-]` dropped.
Audio: **id** = bit-identical, verified flag off vs on; **inv** = identical per request only when the backend is
batch-invariant (check in pedantic mode, see `docs/cuda-serving.md`), otherwise within the self-check tolerance.

| # | item | flag | audio | status |
|---|---|---|---|---|
| L1 | Multi-process load generator, and the client's own CPU in every summary | `pocket_ladder.py --client-procs N` | — | [x] WIN as a tool; ruled the client out (F2) |
| L2 | Stand-in paced server to measure a client's ceiling on any host | `tools/gpu/stub_stream_server.py` | — | [x] done |
| L3 | Device wait vs host share, and a per-call-site sync table, in the serve profile | `MYNAH_SERVE_PROFILE=1` | id | [x] done (F4, F5) |
| L4 | Row ceiling as one build-time constant (`src/row_cap.h`); stamped rebuild; default width buckets extend up to the cap | `make cuda cuda-server ROW_CAP=768` | id | [x] C640 at RTF 0.68 on the L40S (default stays 384) |
| L5 | Confirm that the single-context decoder calls are start-up only | — | — | [x] confirmed: 47,381 calls in two runs (≈ 640 contexts × 74, slot-pool decoder warm-up at the start of the serving loop, ~10 s once). The steady-state syncs are the AR step (11.1 ms), the gang decode (11.7 ms), the survivor flow pass (0.9/iter, 0.41 ms) and the retire drain (5.5/iter, 4 µs) |
| L6 | Deferred context release instead of a full drain on retire: park the context behind an event in the slot pool. Prerequisite for L13 | `MYNAH_CUDA_DEFERRED_RELEASE` | id | [~] coded (backend fence record/wait, `graph_forget_parked`); GPU A/B pending |
| L7 | Ask the sink about cancellation every N iterations (one mutex + `poll()` per stream each time) | `MYNAH_CANCEL_CHECK_EVERY=N` | id | [~] 45/45 WAVs identical on the CPU server; GPU A/B pending; est. 0.5-1 ms/iter at 640 |
| L8 | One `writev()` per HTTP chunk instead of three `send()` calls (writer threads) | `MYNAH_STREAM_OUT_WRITEV=1` | id | [~] 45/45 WAVs identical on the CPU server; GPU A/B pending |
| L9 | FIFO prefill selection stops when a full scan finds nothing (was O(rows²) per step) | none (pure fix) | id | [~] |
| L10 | Duplicate-context checks O(rows²) (~205k pointer compares each, two sites) replaced by a per-context epoch stamp | `MYNAH_DUP_CHECK_EPOCH` | id | [~] coded, GPU A/B pending; S; est. 0.2-0.4 ms/iter |
| L11 | Decoder graph table patch: store each pooled decoder's table column once, scatter the columns into the graph's pinned host tables and `cudaGraphLaunch`, instead of re-recording on every membership change | `MYNAH_CUDA_DECODER_TABLE_PATCH` | id | [~] coded, GPU A/B pending; M; est. 0.7-1.5 ms/iter |
| L12 | Survivor flow reuse: when some rows end, reuse the one-sync flow/latent rows for the survivors instead of `pocket_cuda_flow_step_batch` (saves a sync, a 2.6 MB gather + H2D and a flow pass); fall back when the execution width differs | `MYNAH_CUDA_ONESYNC_SUBSET` | inv (rows already known to end are moved after the survivors in the one-sync flow layout, so under TF32 their last frame can differ in the last bits; reuse only when every ending row was known, the execution width matches and all rows are finite, else today's rerun) | [~] coded, GPU A/B pending; S; est. 0.8-1.5 ms/iter; also removes GPU work, so it helps the L4 |
| L13 | Dispatch-ahead, depth 1: after step k's emit, decode and delivery, plus the prefill pass, queue AR k+1 (`step_launch`), then do retire k, the next admission and cancellation while the GPU runs. An admission joins one step later. Decode k stays synchronous (queuing it with an event is L13b). Needs L6. See "L13 design" below | `MYNAH_CUDA_STEP_OVERLAP` | id at C1 and in the CLI burst (expected); inv under concurrency | [~] coded, untested on GPU. CPU: driver test (burst: same steps, rows and order as the serial loop; continuous admission and a failing row: solo-identical audio) and CLI `--batch 4` WAVs identical with the flag on (launches refused on CPU). Est. +10-25 % (the first cut hides retire + admission + cancellation, not decode or prefill) |
| L14 | Two engines per GPU without code: two servers × 320 rows on two ports, two clients, with and without MPS, against one server at 640 | none | — | [x] 2 × 320: 465 + 448 = **913** audio-s/s, RTF p95 0.68/0.70, GPU **93 %** busy, 300 W (one engine at 640: 799-877, 62 %). With MPS: 908, no gain. Reading: the idle time is between steps (host), not SM underfill; two engines fill it but with less efficient 320-row batches. One engine at ≥ 640 rows without gaps (host diet + L13) is the better target |
| L15 | Decode of frame k on a second CUDA stream, concurrent with AR k+1 | TBD | inv | [-] dropped: MPS showed no concurrency headroom (L14) |
| L16 | Two row groups on two streams inside one engine | TBD | inv | [-] dropped, same reason as L15 |
| L17 | CUDA workers under `--prefork`: open the pack CPU-only and create the CUDA context after the fork | TBD | id | [-] deprioritised: the target is one engine without gaps (larger, more efficient batches; one copy of weights, graphs and slot pool; one admission queue). Two engines were a measurement, not a design. Last resort only |
| L18 | No drain at all when a context is parked | — | — | [-] unsafe: on re-take the host zeroes pinned staging and may free/remap the KV while a late copy could still read it. Implemented as L6 (fence recorded at park, waited at re-take) |
| L19 | Delivery threads: hand the step's PCM batch to N helpers (each stream pinned to one by hash) for the int16 convert + enqueue + `cond_signal`, so the scheduler stops paying one futex wake per stream | `MYNAH_STREAM_DELIVER_THREADS=N` | id on the wire (when on: a queue overflow is noticed by the helper and reported one step later as cancelled; TTFA and sample counters are taken at hand-off) | [~] coded, GPU A/B pending; M; est. 2-4 ms/iter |
| L20 | PCM direct: borrowed pointers into the pinned gang PCM buffer (no 2 memcpys + `calloc`/`free` per row); finite check in the gather kernel | `MYNAH_CUDA_PCM_DIRECT` | id | [~] coded, GPU A/B pending; M; est. 1-2 ms/iter |
| L21 | Lazy hidden: skip the 2.6 MB hidden-state D2H, the finite scan and the per-row 4 KB memcpy; check on the device, copy back only on fallback/dump paths | `MYNAH_CUDA_HIDDEN_LAZY` | id | [~] coded, GPU A/B pending; M; est. 0.5-1 ms/iter |
| L22 | KV table cache: per-context K/V/prefix bases cached at alloc/grow; rewrite a row of the layers × rows metadata only when its context changed | `MYNAH_CUDA_KV_TABLE_CACHE` | id | [~] coded, GPU A/B pending; S-M; est. 0.3-0.8 ms/iter |
| L23 | Draw step N+1's noise (rows × 32 Box-Muller) while blocked in the AR sync; same RNG sequence per request | `MYNAH_CUDA_NOISE_AHEAD` | id | [ ] M; est. 0.4-0.7 ms/iter |
| L24 | Validate decoder op compatibility once per decoder at open, not rows × ops every step | `MYNAH_CUDA_DECODER_VALIDATE_ONCE` | id | [~] coded, GPU A/B pending; S; est. 0.1-0.2 ms/iter |
| L25 | Decoder graphs keyed by width bucket with inert pad decoders (today: exact width, so a new width means capture + instantiate) | `MYNAH_CUDA_DECODER_WIDTH_BUCKETS` | id for real rows | [ ] M-L; tail spikes, not the mean |

Not worth doing (well under 1 µs per row, from the code read):
- the per-row backend-name `strcmp`;
- the linear search over cached decoder graphs;
- KV growth (rare at the initial 256-step capacity).

L10 replaces the old "host diet" umbrella, which is now split into L10, L18-L24.

### L13 design (dispatch-ahead, depth 1)

`MYNAH_CUDA_STEP_OVERLAP=1`, read once when the serving loop starts. Off is today's loop, unchanged.

**Iteration order.**

- Today (flag off), iteration j: admission → cancellation → prefill pass → select → step k (one-sync AR) → emit k →
  gang decode k (sync) → delivery k → retire.
- With the flag, iteration j:
  1. admission, async collect, cancellation. **AR k is running on the GPU during all of this.**
  2. The prefill pass is skipped, because it already ran after the previous step.
  3. Select: exactly the rows queued ahead, in their order.
  4. **Finish** step k (the one sync, then commit) → emit k → gang decode k (sync) → delivery k: all as today.
  5. The **prefill pass** that flag-off runs at the top of iteration j+1, including the late admission.
  6. Select k+1 → **launch** step k+1: queued, no sync.
  7. Retire k.
- The work that hides under AR k+1 (~11 ms at C640): retire k, then admission, async collect, cancellation and the
  loop bookkeeping of the next iteration.
- Not hidden: emit, the decode's host preparation, delivery and prefill.
- First estimate: 5-10 ms of the ~22-25 ms host time per iteration, so +10-25 %. To be measured.

**What moves.**

- The engine gets an optional hook, `step_launch` (appended to the vtable). It runs the queue half of
  `pocket_onesync_step`:
  - H2D of the latents and noise, condition, backbone graph, EOS, the lazy-hidden probe, flow graph, D2H;
  - everything up to the sync, with the same code and the same early noise draw.
- The next `step_batch` on exactly those rows, in that order, runs only the finish half: the sync, the finite gates,
  the commit, the EOS logits and the flow staging.
- Only the prefill pass moves in the driver: after the step instead of at the top of the next iteration.

**The decode does not move (yet).**

- One stream. The gang decode syncs internally (codec transformer, decoder), and delivery k needs it.
- Decode k behind AR k+1 makes its syncs wait for both. Decode k in front of AR k+1 changes nothing, unless the host
  has work that needs neither.
- The larger win needs `decode_audio_batch` split into queue and collect, so the order becomes: decode k + event,
  AR k+1, then admission / cancellation / retire, then the event wait and delivery. That is a follow-up (L13b) once
  L13 is measured.

**No new double buffers. The separation is in time.**

- Emit k reads every AR output (EOS logits, flow output, latent, hidden probe) **before** k+1 is queued.
- At execution, k+1's queued copies and graphs read pinned inputs:
  - the latent input, noise, time embedding and flow cond row table;
  - the KV pointer and position tables.
  These are written at the launch and rewritten only by the next launch or an ordinary step, both of which come after
  k+1's finish.
- PCM: decode k is complete and delivered before the launch.
  - L19 copies the PCM into its helper queue.
  - L20 lends rows owned by the context.
  - AR k+1 writes neither.
- The finish waits with the existing stream sync, so the per-site profile still charges it to `pocket_onesync_step`.
  It also waits for any device work admission queued after k+1 (a voice copy, a decoder warm-up), which would have run
  serially anyway.

**Membership changes take effect at the next launch.**

- EOS, budget and re-prepare are known after emit k, so these rows are left out of launch k+1. Same as flag-off.
- Prefill completions: the pass runs before the launch, so a row that completes there joins k+1. Same as flag-off.
- Admission:
  - The regular pass runs while k+1 is in flight.
  - Those rows are prefilled in the pass after step k+1 and join **k+2**, one step later than flag-off.
  - TTFA is about one iteration longer.
- Cancellation:
  - It is noticed while k+1 is in flight, so the row rides along in k+1.
  - Before that step's emit, the driver marks the row failed. It is stepped and emitted, but not decoded, delivered
    or transitioned.
  - It is then retired as cancelled, one step late.
- Retire:
  - Retire skips a row that is queued ahead. A healthy row never is, because it is active.
  - Because retire's swap-remove runs after the launch, the launch selects over the arrangement that retire will
    produce: the swap-remove is simulated on an index array.
  - So the step gets the same rows **in the same order** that flag-off would select.

**Safety net in the engine.** These entries first discard a step that is queued ahead on their scratch: emit, gang
decode, batched prefill, scratch free, a `step_batch` on other rows, and the `ctx_free` of a queued row. Discarding
means:

1. drain the stream;
2. put back the early noise draws;
3. clear the mark.

Offsets advance only at the finish, so nothing was committed. The next ordinary step recomputes the same frame: the KV
written at the uncommitted position is overwritten with the same values. The driver never takes this path; a counter
and a one-time stderr line say so if it ever does.

**The other flags (all must be on together).**

- **ONE_SYNC** (default on) is required: the launch is its queue half.
  - The launch is refused (and the next step runs as today) in these cases: width < 2, a row that needs KV growth,
    an exhausted step budget, a custom noise function, or one-sync disabled.
  - Refusals are counted.
- **L6 deferred release** is effectively required.
  - Without it, every `ctx_free` drains the stream and waits for k+1. That is correct, but the overlap of that
    iteration is lost.
  - With it, a parked context is fenced after k+1, and a re-take in the same window waits for k+1.
- **L11, L24:** decode only. They run unchanged, before the launch.
- **L12:** the flow row layout is computed at the launch from `eos_step` / `step`. Emit k set those last, as at a
  normal step.
- **L19, L20:** delivery happens before the launch (see above).
- **L21:** emit k settles the lazy hidden rows (copy back or drop) before the launch, so the launch never
  materialises. The finish marks them pending exactly as the one-sync step does today.
- **L22:** the KV tables are written at the launch and read by the graph when it runs. Nothing rewrites them before
  the finish.
- **L7, L8, L9, L10:** unaffected. L10 stamps at the finish's pre-flight.
- **Async admission:** `ctx_attach` runs while k+1 is in flight. It takes no scratch.
- **Overlap stays off** (with a start-up line) in these cases:
  - with the decoder lane;
  - with `MYNAH_PREFILL_SLICE=0`, where admission would run a whole prefill while k+1 is in flight;
  - with parity dumps;
  - for an engine without the hook (CPU builds: Pocket refuses every launch outside the CUDA one-sync path).

**Audio identity.**

- **C1:** id. A lone request steps at width 1, and the one-sync chain, and therefore the launch, needs width ≥ 2.
  What remains is flag-off's loop, with the prefill pass seeing only that request.
- **CLI `--batch 32`:** id expected. The burst meets every condition:
  - every request is admitted in the first iteration;
  - there is no cancellation and no delivery;
  - rows join through prefill (before the launch) and leave on EOS (known before the launch);
  - the launch selects flag-off's rows in flag-off's order;
  - the queued frame is the ordinary one-sync code with its sync deferred.

  The test must show non-zero `launched` in the `[SERVE]` line, otherwise it did not exercise the path.
- **Under concurrency:** inv. Admission and cancellation act one step later, so the composition of a step differs
  from flag-off's, and cuBLAS picks its algorithm by width. This is the same class of difference as any change in
  arrival timing (3b), and it is expected.

**Profile.** `[SERVE] step overlap:` shows steps launched ahead, steps finished from a launch, and refused launches.
The usual host-per-iteration and syncs-per-iteration lines show the effect.

## 3b. Audio identity of the coded flags (2026-10-05, L40S, default production mode)

Concurrent serving is not deterministic, even in pedantic mode: base vs base at C8 matched 164 of 240 WAVs.
Batch composition follows arrival timing, and cuBLAS chooses its algorithm by width. Identity is therefore
checked where batch composition is deterministic:

- **CLI `--synthesize --batch 32`:** one burst, seeds 1000-1031, the batched AR step and retire;
- **the server at C1:** `pocket_ladder` with a fixed seed and deterministic request ids, i.e. the streaming
  path, delivery and the decoder graph.

The base run twice matched 32/32 and 167/167.

| arm | CLI batch 32 | server C1 |
|---|---|---|
| all 11 flags on | 32/32 | 169/169 |
| L6 | 32/32 | 169/169 |
| L7+L8 | 32/32 | 168/168 |
| L10 | 32/32 | 166/166 |
| L11 | 32/32 | 169/169 |
| L12 | 32/32 | 169/169 |
| L19 | 32/32 | 167/167 |
| L20 | 32/32 | 169/169 |
| L21 | 32/32 | 169/169 |
| L22 | 32/32 | 167/167 |
| L24 | 32/32 | 169/169 |

Every coded flag is bit-identical in the production configuration (TF32, bf16 weights, int8 KV) wherever batch
composition is the same. Each flag printed its start-up line in the all-on arm, except L7 and L8, which print
none.

## 3d. Speed A/B of the package (2026-10-05 evening, L40S, ROW_CAP 1024 build, 2-minute levels)

One server per arm, levels C640 → C768 → C896 in that order, `--client-procs 4`. "All" means the 11 coded flags
on (L6, L7, L8, L10, L11, L12, L19 = 4 threads, L20, L21, L22, L24). L9 is always on in this build.

Each cell is audio-s/s / stream RTF p95 / stalls@250:

| C | base r2 | all r1 | all r2 |
|---|---|---|---|
| 640 (first level after start-up, noisy) | 873 / 0.681 / 0 | 983 / 0.611 / 0 | 888 / 0.720 / 0 |
| 768 | 854 / **0.911 ✗** / 0 | **907 / 0.857 ✓** / 0 | **908 / 0.855 ✓** / 0 |
| 896 | 827 / **1.092 ✗** / **46,640** | 985 / 0.836 ✓ / 0 | 960 / 0.960 ✗ / 0 |

The r1 base was lost: the operator's stop command hit its server. The r3 base was not run.

- **Package WIN.** The 0.88 gate moves from C640 (base) to **C768** with all flags on. That is reproducible: 0.857
  and 0.855, with +6 % throughput.
- **C896 with all flags is borderline.** Zero stalls in both runs (the base has 46,640), throughput +16-19 %, but
  RTF p95 0.84 and 0.96 around the gate.
- **GPU busy:** base 63-67 %, all 67-75 %.
- **Host per iteration:** base ~25-29 ms, all ~22-25 ms.
- **Syncs per iteration:** 17-20 → 9.4-9.7. The flags removed the drain syncs and the survivor flow pass.
- **Decoder graph re-records per level:** ~2,000-2,400 → 1-160; table patches ~2,200-2,500.
- **C640 is not reliable.** It is always the first level after start-up and swings 888-983 for the same config.
  Judge on C768 and C896.
- **C768 runs below both C640 and C896 in every arm.** The pattern repeats, so it is not noise: probably a width
  bucket or graph boundary. Open.
- **The single-flag 1-minute screening was stopped:** at 1-minute levels the noise is ±5-10 %. Per-flag
  attribution is to be done by removing one flag at a time from the package.
- **Crash, open:** a ROW_CAP 1024 server (pre-flag code) exited silently between C896 and C1024 in an earlier
  run, with no message in the log. Not reproduced since; not investigated.

Evidence (summaries and job scripts only): `.work/l40s-2026-10-05/` (`res/*.log`, `jobs/*.sh`).

## 3f. 2026-10-06, second L40S box (Intel Xeon Platinum 8558): host too slow to compare

- **Box:** a new Vast.ai L40S on an Intel Xeon Platinum 8558, 4 NUMA nodes, 23-CPU container quota.
  - The cores are capped at 2.1 GHz with the `powersave` governor and no turbo; this cannot be changed from inside the
    container. The 2026-10-05 box was an EPYC 9534 at ~3.7 GHz.
  - The host load average was ~31, so there were other tenants.
  - The serving loop is bound by the single scheduler thread, so host time per iteration about doubled.
- **10-minute soak at C832, unpinned:**

  | arm | audio-s/s | RTF p95 | stalls@250 | SM | host per iteration | syncs per iteration |
  |---|---|---|---|---|---|---|
  | base | 635 | 1.552 | 2,249,005 | 48 % | 60.98 ms | 17.81 |
  | all flags | 699 | 1.450 | 1,185,325 | 52 % | 54.03 ms | 9.18 |

  The package is still a WIN in relative terms on a slow host: +10 % throughput, half the stalls, half the syncs.
- **NUMA pin.** By default the scheduler thread ran on a node other than the GPU's (CPU 170; the GPU is on node 2,
  CPUs 48-71 and 144-167).
  - With the server pinned to the GPU node and the clients on node 0 (2-minute run at C832), the base arm went to
    608 audio-s/s, RTF p95 1.308, 386,984 stalls, host 45.75 ms per iteration (61 ms unpinned).
  - The pinned all-flags arm (`pin-all2`) was lost when the box was destroyed.
  - `knee.sh` now takes `PIN` and `CLIPIN` (taskset lists). Pin the server to the GPU NUMA node on any multi-node
    host.
- **Decision:** move to a higher-clock host, a single-NUMA AMD Threadripper. Leave-one-out at C768 / C896 waits for
  that box.
- **Operator note:** stopping a chain must also kill its running `knee.sh`. An orphaned knee's cleanup
  (`pkill` + `tmux kill-session -t srv`) killed the next arm's server once.
- **Jobs:** `.work/l40s-2026-10-06/jobs/`.

## 3g. 2026-10-06, L4 regression check and leave-one-out (Vast.ai L4, EPYC 7702, ROW_CAP 384)

- **Host check:** no thermal slowdown at 74-80 °C. The clock sits at ~1250 MHz under `sw_power_cap` (72 W), which is
  normal for an L4. The GPU-busy share is 86-90 %, so the L4 is GPU-bound: host time per iteration is ~13-15 ms.
- **Method:** 2-minute levels at C288 and C320, `--client-procs 4`. The reference arms (all flags, base) ran at the
  start and again at the end; drift was within 0-3 audio-s/s.
- Each cell is audio-s/s / stream RTF p95. Stalls and failures were 0 in every arm.

| arm | C288 | C320 | syncs per iteration | host per iteration |
|---|---|---|---|---|
| base | 337 / 0.777 | 341 / 0.847 | 17.95 | 15.08 ms |
| base (repeat) | 339 / 0.771 | 344 / 0.844 | 17.89 | 15.09 ms |
| **all 11 flags** | **351 / 0.748** | **354 / 0.820** | 14.74 | 13.03 ms |
| all (repeat) | 351 / 0.748 | 356 / 0.815 | 14.74 | 13.07 ms |
| without L6 | 349 / 0.751 | 354 / 0.817 | 17.11 | 13.14 ms |
| without L19 | 345 / 0.765 | 351 / 0.826 | 14.87 | 13.92 ms |
| without L21 | 348 / 0.754 | 354 / 0.821 | 14.82 | 13.33 ms |
| without L11 | 351 / 0.747 | 356 / 0.817 | 14.78 | 13.33 ms |
| without L12 | 346 / 0.756 | 351 / 0.824 | 16.30 | 13.19 ms |
| without L20 | 347 / 0.755 | 354 / 0.817 | 14.77 | 13.43 ms |
| without L22 | 351 / 0.747 | 356 / 0.813 | 14.68 | 13.05 ms |
| without L24 | 350 / 0.749 | 355 / 0.816 | 14.73 | 13.08 ms |
| without L10 | 350 / 0.749 | 354 / 0.815 | 14.66 | 12.99 ms |
| without L7 | 348 / 0.754 | 354 / 0.820 | 14.82 | 13.39 ms |
| without L8 | 350 / 0.747 | 355 / 0.819 | 14.68 | 13.07 ms |

- **Package on the L4: a small WIN, no regression.** +4 % audio-s/s and -0.03 RTF p95 at both levels, reproduced.
  Syncs per iteration go from 18 to 14.7. This satisfies the "no regression on the L4" condition of the close-out
  rule (3c) for every flag.
- **Attribution on the L4** (GPU-bound, so these are small effects):
  - The largest single losses come from removing L19 (-6 / -3 audio-s/s, RTF +0.017) and L12 (-5 / -3, +1.6 syncs
    per iteration).
  - Smaller losses: L20, L21, L7 (-3 to -4 at C288).
  - L6: no throughput change, but +2.4 syncs per iteration without it.
  - Neutral on the L4: L11, L22, L24, L10, L8.
  - Their value on the host-bound L40S still has to be measured: the L40S leave-one-out is open.
- **L13 (`MYNAH_CUDA_STEP_OVERLAP`) identity on the L4, all other flags on in both arms:**
  - CLI `--batch 32`: 32/32 identical, with 124 of 128 steps queued ahead.
  - Server C1: 94/94 identical. Only 89 of 9,115 steps were queued ahead, which is expected: width 1 is mostly not
    eligible.
  - No "discarded" line.
- **Cross-tree identity (L13 off).** The morning tree and the L13 tree, all other flags on, are identical:
  - CLI: 32/32 identical audio, with the same sync calls site by site (3,525 per-row decoder syncs in both);
  - server C1: 94/94 identical.

  The flag-off path is unchanged. The lower per-row decoder sync rate seen in the L13 tree's serving profile (2.7
  vs 12.8 per iteration) and its faster start-up (18 s vs 60 s ready) were not explained on the spot. Since audio
  and CLI counts match, they are probably run conditions (page cache, warm-up inside the profiled window).
  **Open:** re-check them on the next box with the same tree for both arms.
- **L13 A/B on the L4** (one repetition, all other flags on in both arms):

  | C | all flags | all + L13 |
  |---|---|---|
  | 288 | 345 / 0.760 / TTFA 132 ms | 350 / 0.750 / TTFA 183 ms |
  | 320 | 351 / 0.824 / 144 ms | 354 / 0.816 / 199 ms |
  | 352 | 355 / **0.891 ✗** / 154 ms | 358 / **0.878 ✓** / 215 ms |

  - 9,774 of 10,000 steps were queued ahead, 3 refused. No errors, no stalls.
  - Throughput +1 %. RTF p95 -0.01, which just moves C352 under the 0.88 gate.
  - **The cost is TTFA p95, +50-60 ms.** That is more than the one iteration expected (~36 ms on an L4): an
    admission waits for the queued step, then for its own.
  - On the GPU-bound L4 the gain is small, as expected. Judge L13 on the host-bound L40S.
  - L13 stays opt-in: the TTFA cost has to be weighed per deployment.
  - **Possible follow-up:** let an admission ride the queued step when a slot is free at launch time, to recover
    part of the TTFA.
- The 255 "stream aborted" lines in `l13-all2/server.log` come from the operator stop of the second repetition
  (clients killed mid-stream). They are not errors.


## 3h. 2026-10-06 evening, Vast.ai L40S (Xeon Gold 6430, 2 NUMA nodes, cores ≤ 2.6 GHz)

**Setup.** The server is pinned to the GPU's NUMA node and the clients run on the other node. GPU at 70 °C, full
2520 MHz, no throttling. Levels are 2 minutes long. Cells are audio-s/s / stream RTF p95 / TTFA p95.

**Package (all 11 flags) vs base:**

| C | base | all 11 flags |
|---|---|---|
| 768 | 773 / 0.919 / 157 ms | 867 / 0.829 / 141 ms |
| 896 | 785 / 1.034, 80,583 stalls | 891 / 0.915 / 158 ms, 0 stalls |

- Syncs per iteration go from 19.5 to 11.8, host per iteration from 33.5 to 28.4 ms.
- The leave-one-out was stopped after 4 flags to save time. The data so far:
  - **L19** carries the most here: removing it costs 5 % (822 / 0.864 at C768).
  - **L21** costs ~2 % when removed.
  - **L6** changes no throughput but saves ~6 syncs per iteration.
- **VRAM** at the end of each arm: 35.0-35.3 GB in every arm, with or without L6. L6 does not grow device memory.

**Defaults decided** (close-out rule 3c; bit-identical, no regression on the L4 or the L40S):
- All 11 flags become default on.
- L19 gets an automatic thread count: 1 at ≤ 4 CPUs, 2 at ≤ 8, 4 above. `MYNAH_STREAM_DELIVER_THREADS=N` overrides
  it and `=0` turns it off.
- To do: implement the defaults, then update `docs/cuda-serving.md` §7 and `configs/perf/`.

**L13 (step overlap) alone: KO as a default.** It stays opt-in, superseded by L13b.

| C | all 11 flags | + L13 |
|---|---|---|
| 768 | 867 / 0.829 / 141 ms | 882-901 / 0.785-0.811 / 186-189 ms |
| 896 | 891 / 0.915 / 158 ms | 881-899 / 0.914-0.928 / 216-220 ms |

- At C768 one run also showed a gap p95 of 872 ms and 209 stalls.

**The silent exit at C1024 is the fd limit, not the engine.**
- The soft `RLIMIT_NOFILE` is 1024. Past ~1000 streams, `accept()` returns EMFILE, the accept loop breaks without a
  log line, and the normal shutdown then answers 503 "server is shutting down".
- With `ulimit -n 65536`, C1024 ran clean: 880 / 1.076, 0 failures.
- The fix is commit `bee72f7`. The server raises its soft limit to the hard one and backs off on EMFILE. On the box
  it logged "open-file limit raised from 1024 to 1048576".
- **Consequence:** C1024 was never really measured before today.

**L13b decode-ahead** (branch `pocket-l13b`, commit `1d244c8`):
- Identity: 32/32 CLI, 32/32 CLI `--stream` and 166/166 server C1 in every arm, including with `DECODE_CHECK`.

| C | all + L13 | all + L13 + L13b | all + L13 + L13b + L13d (+ late wait 3 ms) |
|---|---|---|---|
| 768 | 901 / 0.785 / 186 ms | 979 / 0.727 / 167 ms | 970 / 0.761 / **118 ms** |
| 896 | 899 / 0.914 / 216 ms | **961 / 0.853 ✓** / 195 ms | 947 / 0.899 / 139 ms |
| 1024 | 894 / 1.065 / 246 ms | 943 / 0.998 / 227 ms, 0 stalls | 940 / 1.038 / 160 ms, 10 stalls |

- **L13b passes the gate at C896.**
- L13d trades ~2-4 % throughput for a much lower TTFA.
- Device wait is now ~6 % of the loop and host ~90 % (55-56 ms per iteration at C1024). The GPU-side waits are
  hidden; what is left is pure host time.

**A1a host-context pool** (`MYNAH_CTX_HOST_POOL`, commit `6a5f996`):
- Identity at =0, =1 and =2 (=2 also zeroes the KV): CLI 32/32 and server C1 166/166 (150/150 at =2).
- At C1, 144 of 160 contexts were pooled. The mean context build is 1.28 ms (was 1.4-2.2 ms fresh under load), and
  `codec_setup` goes from 0.5-0.9 ms to 0.03 ms.
- **Speed (all 11 flags + A1a):**

  | C | audio-s/s / RTF p95 / TTFA p95 | SM | power |
  |---|---|---|---|
  | 768 | 1110 / 0.638 / 111 ms | 85 % | 316 W |
  | 896 | 1108 / 0.742 / 128 ms | 83 % | 318 W |
  | **1024** | **1112 / 0.852 ✓ / 147 ms**, 0 stalls | 84 % | 319 W |

- **C1024 passes the 0.88 gate.** Host per iteration drops from ~55 to 26.5 ms, and device wait is back to 52 % of
  the loop.
- `[CTX]` shows the cause: a fresh context built while the server is full costs **~13 ms** (`ar_states` grows with
  load), while a pooled one costs **0.1-0.3 ms**. Admission was the real ceiling.

**`MYNAH_ASYNC_ADMIT` with INLINE=0: 72 "CUDA: out of memory" step failures** at C768 with 32/46 GB used.
- Diagnosis from reading the code: a stale CUDA error from a recoverable failed allocation is re-read by later
  launch checks.
- Fix proposed but not applied: clear the error after a failed allocation in `backend_cuda.cu`.

**Combined tree** (`pocket-combo`, merge `16e4950`): all 11 flags + L13 + L13b + L13d (late wait 3 ms) + A1a.
- Identity: CLI 32/32, CLI `--stream` 32/32.

| C | audio-s/s / RTF p95 / TTFA p95 | SM | power |
|---|---|---|---|
| 896 | 1158 / 0.738 / 111 ms | 89 % | 329 W |
| **1024** | **1157 / 0.846 ✓ / 130 ms**, 0 stalls | 87 % | 327 W |

- **This is the best result.** It is +31 % over the 11 flags alone at C1024 (880 / 1.076), with the lowest TTFA at
  this load.
- The GPU is near its physical ceiling: 87-89 % busy at 327-329 W out of 350 W.

**Ping-pong L26** (`pocket-pingpong`, `6bc7566`, based on L13b, without A1a):
- Identity is good:
  - below the threshold, CLI 32/32;
  - the split (`MIN=2`) vs the reference, 32/32;
  - `DECODE_CHECK` split 32/32;
  - server C1 166/166.
- 45-55 % of host time ran under the other group's GPU work.
- **The speed arm hit "CUDA: out of memory" during the start-up width-bucket warm-up**, with device memory at
  +42.4 GB, and one-sync was disabled. Stopped; no speed numbers.
- Likely cause: group B's scratch is sized at the full `max_batch` (a deferred stage). That doubles the per-width
  scratch at ROW_CAP 1024, on top of the stale-CUDA-error effect seen with async admission.
- **Next:**
  - size group B's scratch at half width (or cap the groups at ROW_CAP/2);
  - clear the CUDA error after recoverable allocation failures;
  - rebase onto A1a;
  - re-test.
- With the GPU already at 87 % on the combined tree, the remaining headroom for ping-pong is small. It matters more
  for slower hosts (4 vCPU).

**Next session:**
1. Make the 11 flags default (L19 with the automatic thread count). Promote A1a and L13b + L13d after one L4
   regression run and a 10-minute soak at C1024 on the combined tree.
2. Merge `pocket-combo` into the main branch.
3. Apply the CUDA stale-error fix, which also covers `MYNAH_ASYNC_ADMIT`.
4. Ping-pong memory fix and re-test.
5. Test on a 4-vCPU host: the combined tree with L19 = 1.
6. Docs: `docs/server.md` (fd limit), `docs/cuda-serving.md` §7, `configs/perf/`.

Evidence (32 KB of summary logs only): `.work/l40s-2026-10-06/res-l40s-jp.tgz`, and the job scripts in
`.work/l40s-2026-10-06/jobs/`.

## 3e. Next session (start here)

1. Soak at the threshold: 10 minutes at C832 (between the solid C768 and the borderline C896), all flags vs base.
2. Remove one flag at a time from the package, the doubtful ones first (L6, then L19, L21), 2-minute levels at
   C768 and C896: what carries the gain? Then mark each item WIN / KO here and in `PLAN.md`, and apply the
   close-out rule (3c).
3. L13 dispatch-ahead: host time is still ~22-25 ms per iteration with the GPU idle. This is the lever for a
   solid C896, and toward C1024. Coded (see "L13 design"); on the box:
   - identity, all 11 flags on in both arms, flag off vs `MYNAH_CUDA_STEP_OVERLAP=1`:
     - CLI `--synthesize --batch 32`, seeds 1000-1031, with `MYNAH_SERVE_PROFILE=1`: expect 32/32. The on arm's
       `[SERVE] step overlap` line must show launched > 0; otherwise the path was not exercised;
     - the server at C1 with `pocket_ladder`, fixed seed: expect all identical. The launch needs width ≥ 2, so this
       arm checks only that the loop around it is unchanged;
   - speed: all flags vs all flags + L13, one server per arm, 2-minute levels C768 / C896 / C1024,
     `--client-procs 4`, server pinned to the GPU NUMA node;
   - report: audio-s/s, RTF p95, TTFA p95 (expect + about one iteration), stalls, `dmon sm`, host per iteration,
     syncs per iteration, launched / refused, and the `pocket_onesync_step` mean wait (should fall);
   - watch stderr for the one-time "step queued ahead was discarded" line. It should never appear.
4. C768 dip: compare the width buckets and the decoder graph widths at 768 vs 640 / 896.
5. The 1024-row crash: reproduce with all flags at C960 / C1024, with a core dump or `compute-sanitizer`.
6. Before any default: one L4 regression knee (C256 / C288 / C320).
7. Box: a Vast.ai L40S, ~30-CPU container quota. Code is unpacked in `/root/mflags` (no git, ROW_CAP 1024 build)
   if the instance survives a stop / start. Otherwise re-provision from this branch (tarball without `build/`,
   `.o` and `.a`, then `make cuda cuda-server CUDA_ARCH=sm_89 ROW_CAP=1024`).

## 3c. Close-out rule (decide per flag once the A/B is in)

- **KO** (no gain, or a regression): decide one of two.
  - Remove the code, if it adds branches nobody will use.
  - Keep it behind its flag, with the reason written here: for example, a gain expected only on a different GPU or
    host shape.
- **WIN** that is all of the following becomes the default:
  - bit-identical (section 3b);
  - safe on both the L4 and the L40S;
  - a gain on at least one of them with no regression on the other (L4 knee C256 / C288 / C320, 2 minutes).

  Either remove the flag entirely, or keep only a rollback switch (`=0`), as the 2026-10-02 defaults did. Then
  update `docs/cuda-serving.md` section 7 and the serving profiles in `configs/perf/`.
- **WIN** that changes behaviour in a visible way stays opt-in and is documented as such. For example, L19
  counts a queue overflow as cancelled.

## 4. Measuring each item

- **Box:** one L40S, 24L, the knee script with `--client-procs 4`, `MYNAH_SERVE_PROFILE=1`, `nvidia-smi dmon`
  per level.
- **Levels:** C384 / 512 / 640 / 768 on the 768-row build. Arms are flag off vs on, same session, same server
  start shape.
- **Report per arm:** audio-s/s, RTF p95, TTFA p95, stalls, GPU busy, device-wait vs host share, syncs per
  iteration.
- **Audio, for id items:** save audio at C8 with a fixed seed. WAVs flag off vs on must be identical
  (`pocket_ladder.py --save-audio`; compare by request id).
- **Audio, for inv items:** the same comparison in pedantic mode (identical), plus the default mode against the
  self-check tolerance and equal audio length per request.
- **Regression:** every WIN also needs one L4 knee (C256 / C288 / C320) with no regression before it becomes a
  default.
