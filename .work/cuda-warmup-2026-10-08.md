# CUDA serving start-up: where the time goes, and the bucket walk (2026-10-08)

Branch `perf-warmup`. Box: Vast.ai NVIDIA L4 (24 GB, sm_89, driver 595), EPYC 7702,
~24.6 CPU cgroup quota, shared with CPU benchmarks under one lock. Builds
`make cuda cuda-server CUDA_ARCH=sm_89 ROW_CAP=<n>`. Server as in
`tools/gpu/knee_closed.sh`: `-w 8 --max-batch N --max-inflight N --max-pending 2048`,
`MYNAH_QUANT_GROUPS=none MYNAH_THREADS=1`, default flags. Scripts:
`.work/l4-2026-10-08/jobs/{startup.sh,ident.sh,cmp.py}`.

Trigger: an L40S `ROW_CAP=4096` 6L build was still starting after 29 min at
45.4 GB VRAM; `ROW_CAP=2048` started; the default L4 build took ~45-57 s.

## What start-up does (server/main.c, after the scheduler thread starts)

1. model open (weights upload) — 0.2-0.4 s.
2. one warm-up request — 0.6-1.9 s.
3. **width-bucket graph walk** (`MYNAH_CUDA_WIDTH_BUCKETS`, default on):
   `--max-inflight` concurrent requests on a 193-token text, request i capped
   at `4 + 2i` steps, so the live width drops by one every two steps from N to 1.
   Cost: ~N^2 row-steps, and every row's KV grows towards the long text (the text
   is 4x `max_tokens_per_chunk`, so past ~375 rows requests run to the
   1500-step budget instead of EOS). At N=320 that is the 44-48 s of the 57 s;
   at N=1024 on the L4 the long caches fill the GPU (+20 GB) and the walk ran
   113 s **capturing no graph at all** (`graphs captured +0`); at N=2048 (6L) it
   did not finish in 14 min (the L40S 4096 symptom, reproduced at half size).
4. `mynah_tts_startup_mark(0)`; then the **slot-pool prefill**: N requests of one
   sentence run to EOS (~75 steps) — 9.5-11 s at 320, 32-95 s at 1024. Its first
   admission runs the A1b re-plan, which frees the walk's long caches and makes
   fixed ones.
5. `mynah_tts_startup_mark(1)`.

The walk exists to capture graphs keyed by the *execution* width, which is the
bucket (`pocket_cuda_exec_width`), plus one per-set decoder graph per request
set. So it only needs to visit each bucket, once on a step where rows retired
(condition input re-staged) and once where none did.

Graph capture itself is cheap: `MYNAH_CUDA_GRAPH_TRACE=1` (added) on the 1024-row
24L start-up shows 2122 captures (2049 of them per-set decoder graphs, key
0x200000) costing 319 ms record + 291 ms instantiate in total. What remains in
the bucket walk is per-set creation at admission (~35-40 ms per set: 320 sets
~11-12 s, 1024 sets ~39-42 s, independent of 6L/24L), i.e. linear in N.

## Change: `MYNAH_CUDA_STARTUP_WALK=1` (opt-in)

- New public `mynah_tts_width_buckets()` (engine's parsed bucket list).
- Bucket walk: same N requests, a 3-sentence text (~45 tokens, ends by EOS well
  after the walk), caps planned so the live width goes N -> N-1 (top bucket's
  retire step) -> each smaller bucket -> 0, two steps each: `4 + 2 x (buckets + 2)`
  steps (~36 at 320). Falls back to the full-width walk when there are no buckets.
- Slot-pool prefill capped at 2 steps per request (the cache is sized at
  admission from the text; what it parks does not depend on how far it runs).
- Always on: one `start-up phases: model open / setup / warm-up / width walk /
  slot-pool prefill; ready after; device memory in use; graphs captured
  (fallbacks), decoder gang graphs, backbone batch max width` line at ready.

## Time to ready and VRAM (L4, `startup.sh`: /health with curl -m 5, nvidia-smi)

| build / model / `--max-batch` | base: ready, VRAM at ready (peak) | `STARTUP_WALK=1`: ready, VRAM at ready (peak) | phases after (warm-up / walk / prefill) |
|---|---|---|---|
| ROW_CAP 384, 24L, 320 | 57.1 s, 16.0 GB (17.0) — walk 44.4 s, prefill 10.8 s | **17.1 s, 11.0 GB (11.0)** | 1.5 / 12.2 / 2.9 s |
| ROW_CAP 384, 6L, 320 | 58.8 s, 10.1 GB (10.3) — walk 48.3 s, prefill 9.5 s | **14.6 s, 5.9 GB** | 0.6 / 11.2 / 2.4 s |
| ROW_CAP 1024, 24L, 1024 | 209.5 s, 22.5 GB (22.5, GPU full) — walk 112.8 s with 0 graphs, prefill 94.7 s | **53.7 s, 21.1 GB**, walk captures 1096 graphs | 1.7 / 42.2 / 9.4 s |
| ROW_CAP 1024, 6L, 1024 | 47.9 s, 15.6 GB (17.9) — walk 15.0 s with 0 graphs, prefill 31.8 s | **48.0 s, 17.3 GB**, walk captures 1096 graphs | 0.7 / 39.0 / 7.9 s |
| ROW_CAP 2048, 6L, 2048 | **not ready after 843 s**, GPU full (22.5 GB) | 21.3 s, 21.0 GB | 0.8 / 3.3 / 16.9 s |
| ROW_CAP 2048, 24L, 2048 | not run (6L already failed) | 7.9 s, 21.7 GB | 1.9 / 3.6 / 2.1 s |

Reading:
- At 1024 rows the base walk was not only slow but useless on 24 GB: it captured
  no graphs (the 6L 1024 base is "fast" for the same reason). The bucket walk
  captures them; its 6L time is therefore not comparable to the base's.
- At 2048 rows on 24 GB neither arm is a working server: the bucket walk finishes,
  but `backbone batch max width 0` (24L) / `graphs captured +0` in the walk (6L)
  show the batched path is not engaged — the per-row buffers do not fit. Only the
  start-up is fixed there; 2048 rows need a 48 GB GPU. The L40S 4096 case is
  expected to behave like the 6L 2048 row above (not re-measured here).
- Start-up now grows linearly with N (~35-40 ms per request set, admission-side),
  instead of with N^2 and with long-text KV.

## Gates

1. **Identity** (docs/benchmarking.md section 3, 24L): base vs `STARTUP_WALK=1`
   (the build had it on by default then): CLI `--batch 32` 32/32, `--stream` 32/32,
   server C1 (`--max-batch 16`) 95/95; base vs base again 32/32, 32/32, 95/95;
   and server C1 at `--max-batch 320` (the walk at full width) 95/95. PASS.
2. **Steady state**, C320, 2-minute levels, fresh server each, interleaved
   (`knee_closed.sh`, PROCS=4, no pinning), audio-s/s / RTF p95 / TTFA p95 / gap p95:

   | run | base | `STARTUP_WALK=1` |
   |---|---|---|
   | 1 | 380 / 0.780 / 121 ms / 125 ms | 377 / 0.786 / 123 ms / 125 ms |
   | 2 | 376 / 0.787 / 123 ms / 126 ms | 377 / 0.786 / 123 ms / 126 ms |

   0 stalls, 0 failures, 97-98 % SM in all. Equal within noise (reference for
   this build family today: 376-386, 0.77-0.79, 120-127 ms). PASS.
3. **First 30 s after ready** (C320 burst at ready, `WARM=0 DUR=30`, one run each):
   base 401 / 0.750 / TTFA p95 693 ms / gap p95 120 ms / 0 stalls;
   bucket walk 391 / 0.770 / **994 ms / 398 ms / 14 stalls**. NOT green: the
   full-width walk warms more than the bucket graphs (likely the decoder-gang and
   first-frame widths, and caches of many sizes) and the first burst pays for it
   with the bucket walk. One sample each; not re-run on request (shared box).

## Decision

`MYNAH_CUDA_STARTUP_WALK` is **opt-in** (unset/0 = the old walk and prefill,
unchanged). Recommended: leave it off for the default L4 build (start-up 57 s is
tolerable and the cold burst is better); turn it on for `ROW_CAP` >= 1024 builds,
where the default walk takes minutes, fills the GPU and at 1024 rows on 24 GB
captures no graphs, or does not finish at all (2048 on 24 GB, 4096 on 48 GB).

## Next

- Close the cold-burst gap so the bucket walk can become the default: after the
  bucket walk, add a cheap pass that visits the decoder-gang / first-frame gang
  widths (or run the first-30-s check with `MYNAH_SERVE_PROFILE=1` to see what is
  captured or allocated during that burst — compare graph_captures at ready and
  after 30 s).
- The remaining linear cost is per-set creation at admission (~35-40 ms/set);
  pre-creating sets off the scheduler thread would cut the 1024-row walk further.
- Re-measure the L40S `ROW_CAP=4096` 6L case with `MYNAH_CUDA_STARTUP_WALK=1`.

## Follow-up: the cold-burst gap closed, bucket walk default on (branch `perf-warmup2`)

Same box and method. Cold burst = fresh server, C320 closed loop for 30 s from the
moment /health answers (`WARM=0`), `/metrics` snapshot at ready and after
(`.work/l4-2026-10-08/jobs/cold.sh`, same ladder flags as `knee_closed.sh`).

### Cause

Not graphs: `MYNAH_CUDA_GRAPH_TRACE=1` after ready showed 74 captures (bucket walk)
vs 93 (full-width walk), all under 3 ms. The `[CTX]` admission counters
(`MYNAH_SERVE_PROFILE=1`) did show it: with the bucket walk the first 640 served
admissions were all "fallback" (malloc 9259 -> 14739, free 8 -> 5447, ~8 device
allocations and frees each) and only then became zero-call; with the full-width
walk they were zero-call from the first. The walk and the prefill capped their
requests with `max_steps` (4..36 and 2), and the per-request buffers (latents,
codec positions, `pocket_ctx_sizes_of`) are sized from `max_steps` at admission,
so every parked set was too small for a served request. The full-width walk's
caps went up to 644 and its prefill ran with the default.

Fix 1: the walk's requests keep the default `max_steps` and are stopped by
cancellation instead (`synth_job.stop_after`, checked in `sink_cancelled`; the
driver polls every 4 steps, so the walk takes ~150 steps at 320 rows, 2.7 s).
Result: zero-call admissions from the start, TTFA p95 693 ms (= base 688), but
gap p95 still ~395 ms and 14 stalls, all on the requests sent ~1.9 s in: one
global ~0.4 s step (the profile's B309 worst 373 ms) about 2 s into the burst,
when the new requests had only one frame buffered. VRAM timeline flat around it
(no allocation). A two-step prefill (no audio decoded) and a one-word prefill
run to its end both kept it; the prefill sentence run to its end (~75 steps,
as the full-width path always did) removed it.

Fix 2: the slot-pool prefill runs to the end in both modes (11 s at 320 rows).

### Gates (24L, ROW_CAP 384, `--max-batch 320` unless noted)

| gate | base (`=0`, full-width walk) | bucket walk (now default) |
|---|---|---|
| identity, CLI `--batch 32` / `--stream` / server C1 (`--max-batch 16`) | reference | 32/32, 32/32, 95/95 identical |
| time to ready, VRAM at ready | 57.5 s, 16.0 GB | 15.7 s, 11.1 GB (walk 2.7 s, prefill 11.2 s) |
| first 30 s of a C320 burst at ready: aps / RTF p95 / TTFA p95 / gap p95 / stalls@250 | 398 / 0.759 / 688 ms / 122 ms / 0; 396 / 0.763 / 680 / 122 / 0 | 391 / 0.772 / 683 ms / 124 ms / 0; 389 / 0.777 / 628 / 125 / 0 |
| C320 2-min steady state: aps / RTF p95 / TTFA p95 / gap p95 | 377 / 0.786 / 123 / 126; 368 / 0.810 / 127 / 130 | 378 / 0.783 / 124 / 126 |

Intermediate (not shipped): fix 1 with a two-step prefill: ready 5.2 s, first
30 s 399 / 0.757 / 693 / 397 / 14 stalls; steady 368 / 0.807 and 374 / 0.791.
Steady-state differences between all arms are run-to-run noise (base itself
377 vs 368).

Not re-measured: `ROW_CAP=1024` (the first opt-in version went 210 -> 54 s; the
full-length prefill adds roughly what it costs at 1024 rows, 30-40 s by the
earlier numbers) and the 6L pack.

Decision: `MYNAH_CUDA_STARTUP_WALK` default on; `=0` restores the full-width walk.

Open: what the prefill run to its end warms that the 2 s step needs (not
graphs, not allocation; a one-word text run to its end does not do it). A
shorter prefill that still covers it would take the 320-row start-up from
16 s to ~5 s.
