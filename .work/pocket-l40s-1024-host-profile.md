# Pocket on one L40S toward C1024: where the host time goes, and how to measure it

Board: `PLAN.md` E15-32 (follow-up to `.work/pocket-l40s-plateau.md`). Branch `pocket-l40s-plateau`. Written
2026-10-06 from a code read only: nothing here was run, and every number marked "est." is an estimate to be confirmed
by the profiler designed in section 2.

**Question.** On an L40S with a 2.6 GHz host, the single scheduler thread spends ~28 ms per iteration on host work
(emit, delivery, retire, admission, prefill bookkeeping) while the GPU is idle (~65 % busy). Before splitting that
work across threads, find out where it goes and what can simply be removed.

**Short answer.**

1. About half of the host time is probably outside `step_live`, and the largest single term is probably **admission**:
   building the request context.
   - At ~900 rows about 8 requests finish and 8 are admitted per iteration. That rate depends only on rows and mean
     request length (~9 s), not on host speed.
   - Each admission costs ~1.4-2.2 ms of host work even with a warm slot pool (`[CTX]` measurement,
     `.work/pocket-cuda-cold-burst.md`).
   - So est. 11-16 ms of the 28 ms.
   - `MYNAH_ASYNC_ADMIT` does not help as configured: its inline threshold (8) covers the whole steady-state
     admission rate.
2. The per-row loops inside the step add up to est. 7-11 µs per row at 2.6 GHz, so 6-10 ms at 900 rows. The decode
   gang's host side is the largest part. Delivery is ~1 µs per row with L19 on.
3. Splitting the per-row loops across N threads would save est. 1.5-3 ms per iteration on a large host, and nothing
   or less than nothing on a 4-vCPU host.
   - It belongs after the cheaper removals in section 4, behind an opt-in thread count.
4. The existing profile cannot confirm any of this:
   - the batched prefill is not accounted;
   - "other" mixes admission, retire, cancellation and the prefill;
   - nothing is per row.

   Section 2 is a `MYNAH_SERVE_PROFILE=2` design that answers it in one run.

---

## 1. qwen-tts tooling and host-side techniques, and what ports to mynah

Source: the author's other engine, `qwen-tts` (read-only). It is CPU-first and the HTTP server is plain C. On CUDA it
is a single process with one sync per step.

What qwen-tts does **not** have, so there is nothing to port for these:
- rdtsc;
- NVTX;
- a Chrome/Perfetto trace exporter;
- flamegraph or sampling helpers (`perf record` is run by hand);
- epoll, writev or eventfd;
- pinned host memory;
- NUMA calls;
- lock-free SPSC rings (every queue is mutex + condvar).

On the host side of CUDA serving mynah is already ahead: dispatch-ahead (L13), one sync per frame, pinned staging,
the slot pool and delivery helpers. What is worth taking from qwen-tts is mostly **observability discipline**.

### 1a. Profiling and observability

| qwen-tts item | what it is | mynah today | port? target | effort | value |
|---|---|---|---|---|---|
| `QWEN_STAGE_TRACE=1` → one `[STAGE] v=1` line per iteration (`qwen_tts.c` ~4428): admit/prefill/head/sample/cp/decode/talker/output/queue_wait ms, `serial_ms` = wall minus named phases, `wall_ms`, `active`, decoder-call counts; CLOCK_MONOTONIC, versioned key=value | per-iteration phase record, explains tail iterations rather than means | `[SERVE]` is end-of-run aggregates only; `sink_phase` (`server/main.c`) has four coarse phases as 5 ms histograms on `/debug/first-chunk` | **yes**, as the windowed `[HOSTP]` line of section 2 (`src/inference.c:serve`) | S on top of section 2 | high: tail iterations (the 390 ms B768 worst case in 3d) become visible |
| `tools/stage_pressure.py` | parses `[STAGE]`, reports phase pressure and the longest iterations; never joins client gaps without an explicit clock alignment | none | **yes**: `tools/host_phases.py` parses `[HOSTP]` lines and the end-of-run table | S | medium |
| `[TTFA2]` / `[PATH] v=2` per-request timeline (client start from a request header, accept, parse, enqueue, admit, prefill start/done, step 1, decode 1, first PCM, write) with `boot_id` to check clock domains | where a request's TTFA went | `sink_phase` + first-chunk histograms (arrival offset, slot full/free, iterations to first chunk) | partial: add `prefill_done_iter` and `first_frame_iter` to `synth_job` and one line per request under level 2 sampling (`server/main.c:record_job_timing`) | S | medium: L13 costs TTFA, this shows where |
| Cost map `QWEN_COST_MAP=1/2` with JSON + `tools/costmap_report.py`: thread-local accumulation, declared nesting checked at run time | inclusive/exclusive regions | **mynah already has the same design** (`src/costmap.c`, `MYNAH_COST_MAP`, `MYNAH_COSTMAP_JSON`) | no: but do NOT reuse it for loop phases. Its regions are compute regions and it is too heavy per row (region stack push/pop per call). Keep it for kernels | — | — |
| `POOLSTATS` dispatch/chunks/park/serial; `qwen_parallel_meter`; pool spin budget `QWEN_POOL_SPIN` | pool overhead | mynah has `MYNAH_POOL_METER`, spin calibration, `mynah_parallel_stats` (`src/threads.h`) | already ported. Relevant only if section 3's parallel-for uses the pool | — | — |
| SIGUSR1 counter dump (prefork stats, pool, census, cost map) | read counters from a live server | mynah: `on_usr1_dump` / `dump_local_stats` (`server/main.c`) | extend: on SIGUSR1 also print the level-2 phase table so far (flag only; printed by the scheduler thread at its next iteration top) | S | medium: 10-min soaks need no restart |
| Prometheus `/metrics` with single-writer relaxed atomics; threshold counters instead of histograms (`ttfa_over_250ms` etc.), justified as exact at any N | live counters | mynah has `/metrics` with sums + counts (`handle_metrics`) | optional: add `mynah_scheduler_host_seconds_total{phase=...}` and `iterations_total`, so host share can be watched live | S | low-medium |
| `[TOPOLOGY] v=1 pid configured_mask actual_mask threads mode` line at start | which CPUs each process actually got | none; `knee.sh` takes `PIN`/`CLIPIN` | **yes**: one start-up line with the affinity mask, the GPU's NUMA node (`/sys/bus/pci/devices/<bdf>/numa_node`), the clocksource, CPU MHz and governor, and the scheduler thread's CPU (`server/main.c:main`) | S | **high**: 3f lost a day to a scheduler thread on the wrong NUMA node and a `powersave` 2.1 GHz host |
| `QWEN_KERNEL_TIMING` / shape census | kernel buckets by batch | dispatch census exists | no (GPU kernels: use nsys) | — | — |
| `tools/costmap_ab.sh` (A/B/B/A overhead check of the profiler itself) | proves the profiler does not move the number it measures | none | **yes** for level 2: the same ABBA with level 1 vs level 2 (section 2e) | S | medium |
| Documented findings (per-region numbers): pool dispatches per frame, context switches/s, `posix_memalign` / mmap / munmap counts per request cut from 11,899 / 78 / 154 to 41 / 1.3 / 1.3 by a decoder arena, bit-identical | the method: count allocations and context switches, not only time | not counted in mynah | **yes**: level 2 also prints `getrusage(RUSAGE_THREAD)` deltas (voluntary/involuntary context switches, minor faults) of the scheduler thread per iteration | S | **high**: admission is suspected to be page-fault bound (section 4) |

### 1b. Host-side serving techniques

| qwen-tts item | measured result there | mynah equivalent | port? (target, effort, value) |
|---|---|---|---|
| Engine thread pool: caller + N−1 workers, one atomic chunk counter, spin then park, need-mask wake, sense-reversing spin barrier for persistent regions | dispatch count was the cost (10.7k dispatches per request); persistent regions cut it 384 → 80 per frame-pair | `src/threads.c` (`mynah_parallel_for`, spin calibration, lanes) is the same family. On the GPU server it runs with `MYNAH_THREADS=1` (`tools/gpu/serve.sh`) | reusable for section 3, but with its **own** width and spin budget, not `MYNAH_THREADS` (M, value depends on the host: section 3) |
| CPU affinity: forked workers pinned core-major (SMT siblings together) from sysfs `core_id`; plans on the inherited mask; `--cpu-mask` | needed for the per-core AMX unit there | none in the server; `knee.sh` uses taskset | **yes**: `MYNAH_SCHED_CPU=<cpu>` (or `auto` = first CPU of the GPU's NUMA node) pins `mynah-sched`; delivery helpers and writers get the rest of that node, never the scheduler's SMT sibling (`server/main.c:scheduler_main`, `stream_out_deliver_init`). S, high on multi-NUMA hosts (3f: 61 → 46 ms host per iteration just from the NUMA pin) |
| Decoder lane (private pinned team) | 11-17 % better than inline at B2-B4, missed its gate | mynah has the lane (E5-21); exclusive with the CUDA decode gang | no (CUDA gang wins) |
| Prefill helper thread | rejected: TTFA p95 435 → 2379 ms | `MYNAH_ASYNC_ADMIT` (context build on helpers) | lesson: moving work off the loop costs TTFA when the work then waits one iteration to be collected. Same trade-off as `MYNAH_ASYNC_ADMIT` (+67 ms TTFA p95 on the L4) and L13 (+50-60 ms) |
| Sliced admission `QWEN_PREFILL_SLICE` | CPU: RTF better, TTFA worse; CUDA: worse | mynah has sliced + batched prefill (default 16 tokens for 24L) | already better in mynah |
| Async output writer: one detached thread per stream, 1 MB bounded queue, `SO_SNDTIMEO`, malloc per chunk, one signal per chunk | neutral (C3 −0.010, C4 +0.009) | the **same design** (`server/stream_out.c`), plus a ring instead of malloc, plus L19 helpers | lesson: at C4 one thread per stream is free; at C900 it is 900 futex wakes and 900 context switches per step (section 4, item H5) |
| Synchronous output: per-thread grow-once int16 buffer, three `write()` per chunk, TCP_NODELAY | default there | mynah: `stream_out_enqueue` (grow-once convert buffer) + L8 `writev` | done |
| Cancellation: per-step `poll(fd, 0)` for POLLRDHUP under a flag | — | `sink_cancelled` → `stream_out_peer_gone` (mutex + `poll`) per row; L7 makes it every N | done (L7); see H4 for a cheaper form |
| Memory: decoder bump arena per stream, grow-once TLS scratch, preallocated batch buffers | allocations per request 11,899 → 41, mmap/munmap 78/154 → 1.3/1.3, bit-identical | slot pool (device + pinned buffers), but the **host** halves (`ar_states`, `codec_setup`) are still allocated per request | **yes**: this is the top candidate (A1 in section 4) |
| Execution budget: one pool owns all CPU work, BLAS held to 1 thread | context switches −48 %, C4 p95 0.98 → 0.95 | `MYNAH_THREADS=1` on the GPU server; the CUDA build links no host BLAS in the hot path | check that no stray thread team exists in the GPU server (`top -H` at C900: expect `mynah-sched`, `mynah-dlvN`, `mynah-outNNN`, HTTP workers, CUDA driver threads) |
| Admission-health publish (seqlock-style release/acquire atomics for `last_iter_ms`) | used by an admission policy that was falsified | — | no |
| Thread naming (`srv-sched`, `srv-output`, ...) | `top -H` answers "which thread is hot" | mynah names `mynah-sched`, `mynah-dlvN`, `mynah-outFD` | done |

**Net:** port the **observability items** (per-iteration line + parser, topology line, rusage deltas, ABBA overhead
check). Port the **memory discipline** to the host halves of the request context. Port the **pinning**.

---

## 2. Design: a fine-grained host-phase profiler (`MYNAH_SERVE_PROFILE=2`)

### 2a. What is missing today

- `[SERVE] loop`:
  - `step` = `step_live` (AR + emit + decode + delivery, device waits included);
  - `other` = everything else;
  - `prefill-first/cont` count **only the scalar prefill path**. The CUDA batched prefill (`prepare_slice_batch`
    returns before the accounting in `src/inference.c:slots_prefill_slice`) lands in `other`. That is why the L40S logs
    show "prefill-first 0.0 % (8 slices)" with hundreds of admissions.
- `[SERVE] device wait` gives host vs device for the whole loop, but not per phase.
- The server's `sink_phase` histograms are coarse (admission / prefill+cancel / step / retire), and on `/debug` only.
- Nothing is per row, per admission or per retirement. Nothing counts allocations, page faults or context switches.

### 2b. Phases

One enum, `mynah_hp_phase`, in a new `src/hostprof.{h,c}` (no dependency on costmap). Each iteration is cut into
contiguous phases by **marks**. A mark closes the current phase and opens the next, so the phases tile the iteration
with no gaps by construction. Device waits are attributed to the phase that is open when they happen.

| id | phase | where the mark goes | unit for "per" column |
|---|---|---|---|
| `ITER_TOP` | lane reap, costmap bookkeeping | `serve`, loop top | — |
| `ADMIT_NEXTJOB` | `sink->next_job` (queue pop, `stream_out_start`: 1 MiB ring malloc, `pthread_create`, setsockopt) | `admit_pass`, around `next_job` | per admission |
| `ADMIT_CTX` | `slot_start` → `ctx_new` (host half), or `async_submit` | `admit_pass` | per admission |
| `ADMIT_ATTACH` | `async_collect` (device half) | `serve` | per collected |
| `CANCEL` | cancellation scan | `serve` | per row scanned |
| `PREFILL_SELECT` | FIFO / round-robin selection | `slots_prefill_slice` | — |
| `PREFILL_RUN` | `prepare_slice_batch` / `prepare_slice` (host + its syncs) | `slots_prefill_slice` | per slice, rows |
| `LATE_ADMIT_WAIT` | `sink->wait_arrival` timed wait. **Blocked, not host** | `prefill_pass` | — |
| `SELECT` | step row selection (or the ahead-row rebuild with L13) | `serve` | — |
| `STEP_PRE` | `pocket_step_batch` pre-flight (dup check, KV reserve) | `src/engine_pocket.c:pocket_step_batch` | per row |
| `STEP_QUEUE` | `pocket_onesync_step` up to the sync: latent staging, noise draw, KV tables, graph launch, flow queue | `pocket_onesync_step` | per row |
| `STEP_SYNC` | the frame sync (device) | same | — |
| `STEP_COMMIT` | commit, finite gates, EOS copy-out | same | per row |
| `EMIT` | `pocket_emit_batch` (+ the survivor flow fallback) | `step_live` | per row |
| `GANG_SELECT` | `stream_gang` pending/ready pass | `stream_gang` | per row |
| `DEC_PRE` | `pocket_decode_audio_batch` up to the first device submission: dup check, `decode_admit`, gang upsample staging, `decode_frame_prepare` loop | `pocket_decode_audio_batch` | per row |
| `DEC_SUBMIT` | Mimi tile / codec transform submit, decoder candidates, `decoder_submit_batch` (L11 table patch), gather | same | per row |
| `DEC_SYNC` | the gang sync (device) | same | — |
| `DEC_POST` | PCM placement, finite scan, `decode_frame_finish` | same | per row |
| `DELIVER` | `slot_deliver` loop → sink callback (enqueue, or L19 hand-off) | `stream_gang` | per row delivered |
| `STEP_TAIL` | eos/reprepare flags, requeue loop | `step_live`, `serve` | — |
| `AHEAD_LAUNCH` | L13: the prefill pass + selection + `step_launch` | `step_ahead_launch` | — |
| `RETIRE_SCAN` | retire predicate scan + swap-remove copies | `serve` | per row scanned |
| `RETIRE_FREE` | `slot_retire`: finalize, `ctx_free` (park/fence, host frees), `on_done` | `slot_retire` | per retirement |

Engine-side marks (`STEP_*`, `DEC_*`) sit behind one global branch, so the engine needs no new vtable entry. When
level 2 is off, `mynah_hp_mark()` is an inline function that tests one static int.

### 2c. Timer

- **x86-64:** `__rdtsc()` (not serialising; enough at phase granularity, ~7-10 ns).
  - Use it only if CPUID 0x80000007 EDX bit 8 (invariant TSC) is set.
  - Calibrate ticks per ns against `CLOCK_MONOTONIC` at loop start and at report time. Report the drift, and refuse
    rdtsc if it is above 0.1 %.
- **aarch64:** `cntvct_el0` with `cntfrq_el0`.
- **Fallback:** `clock_gettime(CLOCK_MONOTONIC)`.
  - It is a ~20-25 ns vDSO call when the clocksource is `tsc`, `kvm-clock`, `arch_sys_counter` or `hyperv_clocksource`.
  - With some cloud clocksources (for example `xen`) it is a **syscall, ~0.5-1 µs**.
  - Print `/sys/devices/system/clocksource/clocksource0/current_clocksource` in the start-up line. The existing
    per-row `now_ms()` in `server/main.c:stream_callback` (the deadline check) then costs up to 1 ms per iteration at
    900 rows (H6).
- Storage per phase:
  - `ticks`, `dev_ticks` (sync time inside the phase), `calls`, `units` (rows, admissions, ...);
  - plus a 32-bucket log2 histogram of per-iteration ticks for each phase that runs once per iteration;
  - all plain `uint64_t`, written only by the scheduler thread.
- Device-wait attribution: `mynah_backend_sync_at` (`src/backend.c`) already times every sync under level 1. At level 2
  it also adds the wait to `dev_ticks[current_phase]`, where `current_phase` is a scheduler-thread global set by the
  marks. Then **host = ticks − dev_ticks** per phase.
- Per-row tail sampling: phases with a per-row callout that can block (`DELIVER`, `CANCEL`, `RETIRE_FREE`) time
  **every 16th row** individually into a log2 histogram, to catch futex wakes and page faults. That is 60 extra timer
  reads per iteration at 900 rows.
- Thread counters: `getrusage(RUSAGE_THREAD)` at loop start and end (Linux) gives voluntary and involuntary context
  switches and minor faults of `mynah-sched`. In the windowed line it is read once per window, not per iteration.

### 2d. Output

1. **End of run, stderr** (with the other `[SERVE]` lines). Example of the intended shape:

```
[SERVE] host phases (MYNAH_SERVE_PROFILE=2, rdtsc 2.60 GHz, drift 0.002%), 9112 iterations, mean live 897.4:
[SERVE]   phase          host ms/it  dev ms/it  p50 ms  p99 ms   max ms   per unit
[SERVE]   admit.ctx          14.10       0.31    13.8    31.2     88.0   1.79 ms/admission (7.9/it)
[SERVE]   dec.post            2.71       0.00     2.7     3.4      9.1   3.02 us/row
[SERVE]   deliver             0.95       0.00     0.9     1.6     12.4   1.06 us/row (p99 row 9.8 us)
[SERVE]   ...
[SERVE]   (sum)              27.9      29.6                               host = loop - device - blocked
[SERVE] scheduler thread: 41.2 vol + 3.1 invol ctx switches/it, 1,830 minor faults/it
```

   - The phase sum must equal the existing `host ... ms per iteration` within 1 %. That is the self-check that the
     marks tile the loop.
2. **Windowed line** (port of qwen `[STAGE]`, aggregated so it does not flood the log). With
   `MYNAH_SERVE_PROFILE_WINDOW=N` (default 1000 iterations), one line per window:
   `[HOSTP] v=1 iter=<first> n=<N> rows=<mean> admits=<sum> retires=<sum> host_ms=<mean> dev_ms=<mean> worst_iter_ms=<max> worst_phase=<name> <phase>=<host ms/it> ...`.
   `tools/host_phases.py` turns it into a time series and flags walking phases (soak drift).
3. **Chrome trace (optional).** With `MYNAH_SERVE_TRACE=/path/trace.json` and `MYNAH_SERVE_TRACE_ITERS=<start>:<count>`
   (default `2000:200`):
   - The scheduler thread appends `{phase, t0, t1, units}` into a preallocated ring of 64k events: 200 iterations ×
     ~25 phases ≈ 5k events, plus per-row samples.
   - It writes Trace Event JSON once the window closes. The write is outside the window and costs one fwrite of
     ~1 MB:
     `{"traceEvents":[{"name":"dec.post","ph":"X","ts":<us>,"dur":<us>,"pid":1,"tid":1,"args":{"rows":897}}, ...]}`
     Counters (`"ph":"C"`) carry live rows and syncs.
   - It opens in Perfetto UI or `chrome://tracing`.
   - Helper threads (`mynah-dlvN`, async admit) can log to per-thread rings under the same window (tid = thread
     index). Phase 2: S.
4. **NVTX (optional build flag).** `make cuda-server NVTX=1` defines `MYNAH_NVTX`. Each mark then also calls
   `nvtxRangePop` and `nvtxRangePushA(name)`, using the header-only NVTX v3 from the CUDA toolkit (no link
   dependency).
   - Under `nsys profile` the host phases line up with the GPU timeline. That shows directly which host phase leaves the
     GPU idle, and whether L13's overlap window is really filled.
   - Effort S once the marks exist. Off by default; it adds nothing to a normal build.

### 2e. Overhead budget and validation

- Budget: **≤ 0.1 % of an iteration** (≤ 50 µs of a ~50 ms iteration) at level 2 without trace.
  - ~30 marks × (one timer read ~10 ns + two adds) ≈ 0.5 µs.
  - Per-row samples: 60 × 2 reads ≈ 1.5 µs.
  - Histogram updates: `__builtin_clzll` + increment, < 1 µs.
  - **Total ≈ 3 µs per iteration.** No allocation, no lock, no syscall in the loop. `getrusage` runs per window.
- Trace mode: + ~20 ns per event inside the window only.
- Validation (the qwen `costmap_ab.sh` idea): ABBA, level 1 vs level 2, same server start shape, C768 2-minute
  levels. Accept if `host ms per iteration` differs by less than the noise, i.e. less than the base vs base
  difference.
  - Gate check: the phase sum equals level 1's `host per iteration`.
  - The L13 arm must show `AHEAD_LAUNCH` > 0, otherwise the run did not exercise the path.
- Env flag: `MYNAH_SERVE_PROFILE=2`.
  - Today any value means level 1: `getenv(...) != NULL` in `src/inference.c:serve`, `src/backend.c:mynah_backend_sync_at`
    and `server/main.c:main`. One helper, `mynah_serve_profile_level()` (0, 1, 2; any non-numeric value = 1), replaces
    the three reads, so existing scripts keep level 1.
- Effort: **M** (1-1.5 days).
  - `src/hostprof.{h,c}` ~250 lines.
  - ~35 marks across `src/inference.c`, `src/engine_pocket.c` and `server/main.c`.
  - The phase attribution in `src/backend.c`.
  - The report.
  - `tools/host_phases.py`.

---

## 3. Parallelising the per-row host work across N threads

### 3a. Which loops are row-parallel

At ~900 rows, per iteration, flags package on (L6-L12, L19-L22, L24):

| loop (file:function) | per-row work | independent per row? | data races / constraints | verdict |
|---|---|---|---|---|
| cancellation scan (`src/inference.c:serve` → `server/main.c:sink_cancelled` → `stream_out_peer_gone`) | atomic load; per-stream mutex + `poll(fd, 0)` syscall, ~1-2 µs | yes: per-job state, per-stream mutex | `synth_assert_scheduler()` asserts the thread; `slots[i]` flags are per row (OK) | **do not thread**: L7 already divides it by N. Better still, read the writer's `dead` atomic every iteration and `poll` only every N (H4). Lock-free, ~5 ns per row |
| one-sync staging + noise draw (`pocket_onesync_step`) | 128 B memcpy + 32 Box-Muller normals ≈ 1 µs | yes: each context owns its RNG | the early-draw put-back must stay per row (it does) | prefer **L23** (draw while blocked in the previous sync). Free time, no threads |
| step pre-flight / KV tables (`pocket_step_batch`, `pocket_cuda_backbone_step_batch_impl`) | ~0.3-0.5 µs with L10/L22 | yes (writes `scratch->...[i]`) | `pocket_cuda_backbone_reserve` may allocate device memory (must stay serial; rare) | no: too small |
| emit (`pocket_emit_batch`) | ~0.3-0.5 µs | yes | `scratch->flow_*[gathered]` uses a running index (a prefix sum is needed); region calls go to costmap thread-local stacks | no: too small |
| decode pre (`pocket_decode_audio_batch`: dup check, `decode_admit`, `pocket_cuda_codec_gang_upsample` staging, `pocket_decode_frame_prepare`) | ~1-1.5 µs | mostly. The dup check is O(n²) **across** rows | `pocket_decode_batch_drop` writes a shared `reported` flag and `error`. `mynah_backend_note_*` counters are plain ints. Fallback branches call the backend (`h2d`, `transform_one`). Device calls must stay on one thread, in order | partial: split into a pure-host per-row part (parallel) and a serial device part. Remove the dup check instead (A3) |
| decoder submit (`pocket_cuda_decoder_submit_batch`, L11 table patch in `gpu/cuda/backend_cuda.cu:decoder_step_batch_impl`) | validation + table columns ~0.7-1.5 µs | the column scatter is row-parallel | inside the backend, writes the pinned tables of one graph (disjoint columns: safe) | maybe, inside the backend only. Low priority |
| decode post (PCM placement, `pocket_all_finite` on 1920 floats, `pocket_decode_frame_finish`) | ~1.5-2.5 µs | yes | same `reported`/`error` and note counters | **best candidate** if threading at all. But the finite scan can first be made ~4-8× cheaper (A4) |
| delivery (`stream_gang` → `slot_deliver` → `stream_callback`) | L19 on: ~1 µs (two lane mutexes + a 7.7 KB copy); L19 off: 3-7 µs (convert, ring copy, futex wake) | yes, per stream | `synth_assert_scheduler()`. Per-stream order must be preserved (L19 pins a stream to one helper). The callback updates `sink->first_audio_*` and `audio_samples` (per job: OK) | already offloaded by **L19**. Next: batch the hand-off per lane per step (H5) |
| retire (`serve` retire loop, `slot_retire`) | ~8 retirements per iteration | no: swap-remove mutates the array; `on_done` order | — | do not thread. Offload host frees instead (A2) |
| admission (`admit_pass` → `slot_start`) | ~8 × 1.4-2.2 ms | per request | — | the existing helpers (`MYNAH_ASYNC_ADMIT`) are the threaded form. See A1 |

### 3b. Expected gain

- **Row-parallel host work** that is not better removed first: decode pre + post + submit columns + emit ≈ 3-4.5 µs
  per row. At 900 rows, 2.6 GHz, that is est. **2.7-4 ms per iteration**.
- **Overhead per parallel region.**
  - Fork/join on a spinning pool: ~2-5 µs. With parked workers (futex wake): ~20-50 µs. There would be ~3 regions
    per iteration.
  - The work stays memory-bound on the contexts: each row touches its own heap context. After a worker touches it,
    the scheduler thread's next pass over the same context takes coherence misses (~50-100 ns per line). That claws back
    part of the gain.
- **Large host** (≥ 16 CPUs on the GPU's NUMA node), T = 4 workers pinned to that node, not on the scheduler's SMT
  sibling, spinning ≤ 20 µs between regions:
  - saving ≈ 3.3 ms × (1 − 1/4) − 3 × 5 µs − coherence ≈ **1.5-2.5 ms per iteration**;
  - that is ~6-9 % of the 28 ms, so est. +3-5 % throughput while host-bound. With L13 hiding part of the host time,
    the visible gain shrinks further.
- **4-vCPU host (2 cores + SMT):**
  - The scheduler already shares its 4 hyperthreads with the 4 L19 helpers, ~900 writer threads woken every step, the
    HTTP workers, the CUDA driver threads, async-admit helpers if on, and the load client on the same host.
  - A worker on the scheduler's SMT sibling makes both run at ~0.6× speed.
  - Spinning workers steal exactly the cycles the writers and the client need. Parked workers cost a futex wake per
    region, comparable to the work.
  - **Expected: zero to negative.** F3 already showed that the 4-vCPU host was CPU-starved (C640: +17-22 % just from
    more CPUs).

### 3c. If it is built: shape and gating

- `MYNAH_SERVE_HOST_THREADS=N` (default 0 = off). At start, refuse it with a start-up line unless all of these hold:
  - the affinity mask holds ≥ 8 CPUs;
  - N ≤ (CPUs on the GPU node − 2 − delivery helpers);
  - one-sync is on.
- **A private team**, not `mynah_parallel_for`.
  - The GPU server runs the engine pool at `MYNAH_THREADS=1`. Its spin calibration was tuned for kernels, not for
    ~1 ms host regions.
  - The team is pinned to the GPU node: first the CPUs that are not SMT siblings of `mynah-sched`.
  - It spins ≤ 20 µs, then parks.
- Used only when rows ≥ `MYNAH_SERVE_HOST_PAR_MIN_ROWS` (default 256).
- Contract per loop:
  - a pure-host, per-row body that writes only its own context and a per-row result slot (error string, failed flag,
    "note" increments);
  - followed by a serial merge on the scheduler thread, in row order: first error reported, backend notes summed,
    device calls issued.
  - Because the merge runs in row order and the device work is issued from the scheduler thread in the same order, the
    audio stays **id**. The bodies do not touch any shared float state.
- Effort: **M-L** (2-3 days). The `pocket_decode_audio_batch` split into host/serial halves is the bulk of it, and it
  must keep the CPU/compatibility schedule untouched.
- Value: **low-medium, host-dependent.** Do it after section 4's A-items and L13 have been measured, and only if
  level 2 then shows `dec.pre` + `dec.post` above ~3 ms per iteration on the target host.

---

## 4. Where the ~28 ms goes at ~900 rows (first-principles estimate)

### Assumptions

- 2.6 GHz host, flags package on, L13 off, ~900 live rows.
- Pocket 12.5 Hz, `audio_emit_frames` = 1, so every row delivers one 1920-sample frame per step.
- Mean request ~9 s of audio (from ~5.5 completions per iteration at C640). That gives ~8 admissions and ~8
  retirements per iteration.
- Cross-check against the 10-05 L40S logs (3.7 GHz host, all flags, C640/C768):
  - iteration ~52 ms;
  - `step` 75 % (~39 ms, of which ~29 ms device wait, so ~10.6 ms host inside `step_live`);
  - `other` 24-25 % (~12.6 ms: admission, cancellation, prefill incl. its device time, retire);
  - host 23-25 ms per iteration.

  At 900 rows and a 1.4× slower host, ~28 ms is consistent with ~11-12 ms inside the step and ~15-17 ms outside it.

### Estimate per phase

| phase | per-unit cost (est., 2.6 GHz) | units per iteration | ms per iteration (est.) | basis |
|---|---|---|---|---|
| **admission: context build** (`slot_start` → `pocket_ctx_create`: `ar_states` + `codec_setup` + device) | 1.4-2.2 ms (measured on a warm pool) | ~8 | **11-16** | `[CTX]` breakdown: `ar_states` 0.45-0.97, `codec_setup` 0.53-0.87, device 0.13-0.35 ms; ~90 % host. Likely page-fault/memset bound (MBs of host state that device-owned rows never read) |
| admission: `next_job` (`stream_out_start`: 1 MiB ring, `pthread_create`, setsockopt) | ~0.11 ms | ~8 | ~0.9 | measured `[ADM]` |
| prefill pass (batched slices: host staging + launches; its syncs are device time) | 0.3-1 ms per call, host | 1-2 | 0.5-1.5 | `prepare_slice_batch`; not accounted today |
| retire: `ctx_free` (fence, park, ~25 host frees incl. large buffers → munmap + TLB shootdown across a ~1000-thread process), `on_done`, stats | 50-200 µs | ~8 | 0.5-1.5 | code read |
| retire scan + swap-remove (`synth_slot` ~500 B copy) | ~2 ns per row | 900 | <0.01 | |
| cancellation (L7, every 4) | 1-2 µs per row / 4 | 900 | 0.25-0.5 | `poll` + mutex per stream |
| step pre-flight + KV metadata (L10, L22) | 0.3-0.5 µs per row | 900 | 0.3-0.5 | |
| one-sync staging + noise draw | ~1-1.3 µs per row | 900 | 0.9-1.2 | 16 Box-Muller pairs (log, sqrt, sincos) per row |
| commit + finite gates + EOS copy | ~0.2 µs per row | 900 | ~0.2 | |
| emit | 0.3-0.5 µs per row | 900 | 0.3-0.5 | region calls, flags, two 128 B copies |
| gang selection | ~0.1 µs per row | 900 | ~0.1 | |
| **decode, host side** | | | **3.5-5** | |
| — dup check, O(n²) | n/2 compares × ~0.5 ns | 900 | ~0.2-0.25 | `pocket_decode_audio_batch`: **not** covered by L10's epoch stamp |
| — admit, upsample staging (denorm 32 floats), `decode_frame_prepare` | ~0.8-1.2 µs per row | 900 | 0.7-1.1 | |
| — Mimi tile tables, decoder candidates, submit validation, L11 column scatter | ~0.8-1.5 µs per row | 900 | 0.7-1.4 | |
| — post-sync: 7.7 KB PCM copy (L20), finite scan of 1920 floats, `decode_frame_finish` | ~1.7-2.5 µs per row | 900 | 1.5-2.3 | the scan has an early exit per element, so it does not vectorise; CUDA host objects build at `-O2` without `-march` |
| **delivery** (L19 on) | ~0.9-1.3 µs per row (deadline `now_ms`, 2 lane mutexes, 7.7 KB copy, rare wake) | 900 | **0.8-1.2** | `stream_out_deliver`. With L19 off: 3-7 µs per row = 2.7-6.3 ms |
| fixed per step (graph launches, cuBLAS/tile launches, H2D/D2H issue, sync calls) | — | — | 0.5-1.5 | ~10-30 API calls at 2-5 µs |
| cache misses on the per-row passes (~8 passes over 900 heap contexts) | 0.5-2 µs per row | 900 | 0.5-1.8 | contexts are large and spread out; L3-resident at best |
| **total** | | | **~22-33** | brackets the observed ~28 ms |

### Not on the scheduler thread, but competing with it for CPUs

- The ~900 writer threads (`mynah-outFD`) are woken once per step each: 900 futex wakes, 900 context switches, and
  900 `writev`s of ~3.9 KB into loopback TCP every ~60 ms.
- That is est. 5-10 µs of CPU each, so **4.5-9 ms of CPU per iteration**, plus the same order for the client
  receiving it on the same host.
- On a 4-vCPU host this, not the scheduler's own code, is the likely reason F3 found the host CPU part of the ceiling.

### Top candidates to cut, in order of expected ms per iteration per unit of effort

Each candidate is an A/B flag, default off until measured, under the 3c close-out rule of the plateau note.

**A1. Cheap admission** (est. **−8 to −13 ms**). Pick one of three ways, best first.

- **(a) Pool the host halves of the context in the slot pool.**
  - Targets: `src/engine_pocket.c:pocket_ctx_create` / `pocket_cuda_slot_park` / `pocket_ctx_free`.
  - The parked set keeps the backbone and codec `transformer_ar` host states, the SEANet/codec host state, and the
    `call` scratch.
  - On take, reset by the same zeroing the fresh path implies. Better: do not allocate the host KV mirror at all for
    device-owned rows (lazy-allocation mode in `src/transformer_ar.c`, already named as missing in
    `.work/pocket-cuda-slot-pool.md`).
  - id if the reset is exact. Prove it with the slot-pool leak check (same seed and text right after another
    request).
  - Effort M. No TTFA cost.
- **(b) Quick measurement first:** `MYNAH_ASYNC_ADMIT=2 MYNAH_ASYNC_ADMIT_INLINE=0..2`, a one-line config change.
  - The default inline threshold of 8 equals the steady-state admission rate at 900 rows, so today every admission is
    built on the scheduler thread.
  - Costs one iteration of TTFA (+67 ms p95 measured on the L4).
  - Large host only: the helpers contend on 4 vCPUs (context build rose 2.2 → 3.1 ms there).
- **(c) With L13 on**, admission runs while AR k+1 is in flight (~11-13 ms), so (a) and L13 compound. Admission alone
  (~11-16 ms) is larger than the window L13 opens, so without (a) the overlap is still not enough.
- **First confirm the term** with level 2 (`admit.ctx` ms per admission, and minor faults per iteration of the
  scheduler thread).

**A2. Reaper for host frees** (est. −0.3 to −1 ms).
- `pocket_ctx_free`'s host `free`s and `stream_out` teardown (1 MiB ring munmap, writer thread exit) go to a helper
  thread through a list. The device parts stay as they are (L6 fence).
- The large `free`s → `munmap` → TLB-shootdown IPIs hit every CPU the process runs on. With ~1000 threads that is all
  of them.
- Also allocate the 1 MiB stream ring from a recycled pool (`server/stream_out.c:stream_out_start`), which removes an
  mmap/munmap pair per request.
- id. Effort S-M.

**A3. Epoch-stamp the decode-gang duplicate check** (est. −0.2 ms, grows as n²: −0.3 ms at 1024).
- `src/engine_pocket.c:pocket_decode_audio_batch` still runs the O(n²) loop. L10 covers only `pocket_step_batch`.
- Reuse `MYNAH_DUP_CHECK_EPOCH`. id. Effort S.

**A4. Branch-free finite scans** (est. −0.5 to −1 ms).
- `pocket_all_finite` exits early per element, so it cannot vectorise.
- An OR-reduction of `(bits & 0x7f800000) == 0x7f800000` over the row, with one test at the end, vectorises even at
  `-O2`.
- Better still: L20's gather kernel computes the per-row flag on the device (the plateau note says L20 does this; the
  host scan in the post-sync loop is still there). Then the host scan of 1920 floats per row goes away.
- Same treatment for the 2 × 32-float scans in the one-sync commit.
- id (same predicate). Effort S.

**A5. L23 noise ahead** (est. −0.9 to −1.2 ms at 900 rows; the plateau note's 0.4-0.7 ms was for 640 rows on a faster
host). Already planned; the draw moves into the shadow of the previous sync. id.

**A6. One timestamp per step for the deadline check** (est. −0.02 to −1 ms, clocksource-dependent).
- `server/main.c:stream_callback` calls `now_ms()` per row.
- Pass a per-step "now" through the sink (or cache it in a scheduler-thread global set at the `sink_phase`
  boundary).
- Check the clocksource first: on a syscall clocksource this is ~1 ms per iteration. id. Effort S.

**A7. Batched L19 hand-off** (est. −0.3 to −0.6 ms).
- `stream_out_deliver` takes the lane mutex twice per row: once for the spare pop, once for the append.
- Instead, collect the step's items per lane in a scheduler-local list and splice each lane once per step: 4 lock
  pairs instead of 1800.
- Also keep the items' PCM capacity at one frame from the start (no `realloc` on warm-up). id. Effort S.

**A8. Lock-free cancellation** (est. −0.2 to −0.4 ms with L7=4).
- `sink_cancelled` reads `out->dead` (atomic, already maintained) and `j->gave_up` every iteration. It calls the
  mutex + `poll` probe only every N iterations, since the writer usually finds the hangup first anyway.
- id (same outcomes, noticed at most N−1 frames later, as with L7). Effort S.

**A9. Writer model** (CPU budget, not scheduler time; est. frees 4-9 ms of CPU per iteration on the host). Medium
term.
- Replace one writer thread per stream with the L19 helpers writing directly to non-blocking sockets.
- Each stream gets a small pending buffer; `epoll` is used only for EAGAIN, and one `writev` is made per stream per
  step.
- This removes ~900 wakes and context switches per step and ~900 thread stacks.
- Effort L (`server/stream_out.c`, the cancellation and backpressure contract must be kept). Value: high on small
  hosts, relevant for C1024 anywhere.

**A10. Pinning and topology line.**
- `mynah-sched` goes on the GPU's NUMA node, away from SMT siblings of busy threads. Print the clocksource, MHz and
  governor at start-up.
- 3f measured 61 → 46 ms host per iteration from the NUMA pin alone. Effort S.

**Then L13** (dispatch-ahead): it hides retire, admission and cancellation behind AR k+1.

**Then, only if level 2 still shows decode pre/post above ~3 ms per iteration on a large host:** the opt-in host thread
team of section 3c.

### Rough total

- A1(a) + A2-A8 remove est. **11-19 ms** of the ~28 ms: host per iteration ~10-17 ms at 900 rows.
- With L13 hiding a further part, the GPU-busy share should move from ~65 % toward 80-90 %. Those are the conditions
  under which C1024 at RTF p95 < 0.88 becomes plausible on one L40S.
- All of this is an estimate. The first job on the next box is one level-2 run at C896 that replaces this table with
  measured numbers.

## 5. Next steps

1. Implement `MYNAH_SERVE_PROFILE=2` (section 2) and the topology line (A10). ABBA the overhead on the Mac CPU server,
   then on the box.
2. One box run, all flags, C896, level 2 + `MYNAH_SERVE_TRACE` window + `nsys` with `NVTX=1` for 30 s. Replace the
   section 4 table with measured numbers.
3. A1(b) as a config-only arm (async admit, inline 0) at C896 to bound A1's value before writing A1(a).
4. A3, A4, A6, A7, A8 (all S, all id) as one package with leave-one-out, then A1(a), A2, L23.
5. Re-measure with L13; decide on section 3 only then.
