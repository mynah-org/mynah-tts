# PocketTTS 24L on CPU: streaming capacity on Axion

Status: **open** · sessions 2026-09-25 and 2026-09-26 · **C64 passes every
mandatory gate over 30 minutes**, two preferred gates miss by a hair · board
item `PLAN.md` E15-26.

Every number below was measured on the target host. A screen never promotes.

## Where we stand (read this first)

**Current 24L baseline** -- 30-minute soak, 28328 requests, 2026-09-26:

| field | value |
|---|---|
| model | Pocket 24L, official English `english_2026-04_24l`, rev `492522650173a0653b7575cdc25ae09810e5d741` |
| host | GCP Axion `c4a-highcpu-32`, 32x Neoverse-V2, 62 GB, `BLAS=none`, `SIMD=auto`, gcc 15.2 |
| source | branch `pocket-24l-cpu-segments` = `bea336c` + text segmentation (uncommitted when measured, worktree `scratchpad/wt-24l`) |
| topology | `--prefork 16 --prefork-threads 2 --max-batch 8` |
| backbone | **int8** (`MYNAH_QUANT_GROUPS=codec_transformer:int8,codec_conv:int8,codec_convtr:int8,backbone:int8,flow_net:f16,conditioner:f16`) |
| segmentation | `MYNAH_POCKET_SEGMENT_TOKENS=50`, `MYNAH_POCKET_FIRST_SEGMENT_TOKENS=24` |
| prefill | `MYNAH_PREFILL_SLICE=16` (one tile), `MYNAH_PREFILL_STEP_MS` unset |
| concurrency | **C64** |
| TTFA p95 | 341 ms (long 406, medium 331, conversational 296, short 221) |
| TTFB p95 | 102.8 ms -- **FAIL** by 2.8 ms against 100 |
| STREAM_RTF p50/p95 | 0.711 / 0.780, flat over ten 180 s windows (drift +0.009) |
| required prebuffer p95 | 41 ms; safe_play_start p95 349 ms |
| stall@500 | **0 of 28328** |
| stall@250 | 20 of 28328 (0.07%) -- **FAIL** against zero |
| max_gap p95 | 404 ms (the segment-transition bubble; see finding 8) |
| throughput | 82.8 audio-s/s |
| serving step | B4 51.3 ms (mode, 341082 frames), B3 40.9 ms; mean_live 3.88 |
| verdict | MARGINAL on the gates; NOT QUOTABLE for cadence percentiles (client coalesced 15.5%, refusal at 15%) |

Evidence: `reports/20260926-24l-axion/soak64-30min-20260926-1135.log` (and a
copy in `~/p24-evidence/` on the box's disk).

Compared with the 6L small pack on this host (C120/C126 GOOD, ~155 audio-s/s):
24L now sustains **C64 at 0.53x the 6L throughput**, up from ~C40-48 and 0.44x
on 2026-09-25. The KPI for 24L remains
`audio-s/s per core + Cmax GOOD + TTFA p95 + zero stall@500`.

**Quality:** int8 backbone and segmentation both pass an ASR gate (finding 9)
and a listening check by the owner ("audio molto buoni, ottima qualità") on
2026-09-26. Segmentation is still opt-in (env) and not the shipped default.

## Findings

1. **Ceiling.** 16x2 bf16 saturates at ~67-69 audio-s/s: C64 -> C80 adds no
   throughput, only queue (RTF p95 0.974 -> 1.230).
2. **int8 backbone beats bf16 on 24L in serving**, unlike on 6L (where it was
   rejected on 2026-09-20). C48: +16% throughput at equal concurrency, B1 step
   27.5 -> 16.7 ms (-39%), B3 49.8 -> 44.0 ms (-12%), TTFA p95 830 -> 752,
   stall@250 30 -> 11. It also improves C56, but C56 is not stream-safe on
   either (stall@500 8 int8 / 17 bf16). The CLI shows why: bf16
   `T(B) ≈ 18 + 4.4 B` ms, int8 `8.8 + 5.7 B` -- on 24L the per-step weight pass
   is big enough that halving its bytes matters; the advantage vanishes by B4.
3. **`MYNAH_PREFILL_STEP_MS` is inert on 24L.** 20/30/40/60 ms are
   indistinguishable from the repeated control. A step always runs at least one
   slice (`src/inference.c:306-308`) and one 24L slice already exceeds any cap
   tried.
4. **Prefill slice: a TILE, not a plateau.** `pocket_text_flush_limited`
   rounds the slice down to a multiple of `TAR_PREFILL_TILE = 16`
   (`src/transformer_ar.c:97`) and raises a zero result to one tile
   (`src/engine_pocket.c`, "never make no progress"). So `MYNAH_PREFILL_SLICE`
   8, 16 and 24 all execute **16 tokens**, which is why they measured
   identical; 32 is two tiles and 64 four.
   - **CORRECTION OF THE CORRECTION, keep both:** on 2026-09-25 the afternoon
     fit `gap ≈ 59 + 4.7 ms/token` was declared wrong because 8/16/24 matched.
     It was the correction that was wrong: those three points are one point.
     With the step at ~50 ms, max_gap p95 134 / 209 / 354 ms for 16 / 32 / 64
     tokens gives **~5 ms per prefill token** on two threads, ~85 ms for the
     smallest possible slice -- more than the ~30 ms of slack per frame. Token
     slicing cannot go below one tile; a smaller freeze needs layer-range
     slices.
5. **Why slicing cannot fix TTFA.** At C48 a step costs ~44-50 ms of an 80 ms
   frame, so each worker has ~30 ms of slack per frame. A long text's prefill
   therefore spans many frames whichever way it is sliced: slicing chooses who
   pays (first audio vs continuity), it does not reduce the work in front of
   first audio. Only removing work from the critical path fixes TTFA.
6. At C1 the 24L costs almost nothing extra: CLI RTF 0.41 (24L) vs 0.385 (6L),
   2 threads. The cost appears under batching.
7. **Text segmentation is the lever that fixed TTFA (2026-09-26).** Upstream
   Pocket splits long input (`split_into_best_sentences`, `max_tokens=50`) and
   generates each chunk as its own utterance; mynah prefilled the whole text in
   front of the first audio. Implemented on branch `pocket-24l-cpu-segments`:
   `src/text_segment.c` (the split, above the engine, where the text still
   exists), `mynah_tts_request.segment_lengths`, a `reprepare` step result, and
   a per-segment prologue in `engine_pocket.c` that restarts the backbone from
   the voice prefix while the frame history and the codec run straight
   through. The driver re-queues the slot behind every waiting prefill. At C48,
   int8, slice 16: TTFA p95 743 -> **275 ms**, long 1224 -> 379 ms, stall@250
   9 -> **0**, RTF p95 0.653 -> 0.595, throughput 67.1 -> **73.7 audio-s/s**,
   and the modal step 42.9 -> 39.2 ms (shorter KV per segment).
   `MYNAH_POCKET_FIRST_SEGMENT_TOKENS=24` (a tighter first chunk) is worth
   another ~55 ms of TTFA p95 over plain 50.
   - Correctness: split identical to upstream on **277/277** bank texts (34
     multi-segment); with segmentation OFF the WAV is **byte-identical** to
     `bea336c` on both 6L and 24L; at temperature 0 the last segment of a
     segmented request is **bit-identical** (hidden state, EOS logits, 107 steps)
     to the same sentence generated alone.
   - Deliberate divergence from upstream: upstream opens a new Mimi state per
     chunk; here the codec runs through the boundary. Listened to: no audible
     boundary artifact. Segmentation adds ~0.6-0.8 s per long text (pauses at
     boundaries); it does NOT recover skipped words -- unsegmented long texts
     were already transcribed at 0.8% WER, so the E2-5 "skips a quarter of the
     speech" warning does not show on this 24L pack at ~85-100 tokens.
8. **The next visible bubble is the segment transition.** max_gap p95 rises to
   355-425 ms with segmentation (from ~125): a stream pauses while its next
   segment is prefilled behind the other slots. Zero stalls@500 up to C64
   because the client already holds lead, but it is the first suspect above
   C64. Preparing segment N+1 early only moves prefill work, it does not remove
   it; it matters only if a profile shows the bubble, not the ceiling, is what
   stalls.
9. **ASR quality gate (nemotron-3.5 streaming 0.6b, offline, int8).** The
   offline ASR DROPS THE TAIL of ~20 s inputs with pauses: the "missing last
   clause" first seen on segmented audio was the recognizer (the last 9 s alone
   transcribe correctly; the streaming mode drops it too). The gate therefore
   cuts every WAV at >=250 ms pauses into <=12 s pieces for every arm alike.
   Seeds 1-4, 33 long texts (8200 words) and 30 medium (1424 words):

   | arm | WER long | WER medium |
   |---|---|---|
   | f32 backbone | 0.87% | 0.49% |
   | bf16 | 0.73% | 0.56% |
   | int8 | 0.83% | 0.56% |
   | bf16 + seg50 | 1.12% | 0.56% |
   | int8 + seg50 | 0.94% | 0.56% |

   int8 is indistinguishable from f32/bf16; segmentation costs 0.1-0.4 points
   on long texts, within or just above the f32-vs-bf16 spread. Medium texts
   are one segment, so their seg arms equal the unsegmented ones exactly.

## Closed dead ends (do not repeat without new evidence)

- **4x8** (`--max-batch 16`, needed for 64 slots): 31.5 audio-s/s vs ~69 at 16x2,
  C48 NOT STREAMABLE (RTF p95 1.559, 71% stall@500).
- **32x1**: worse on every column at C48 (TTFA p95 923, 17 stalls@500, max_gap
  303 ms) and the same-host client coalesced 35%.
- **Slice below one tile (16 tokens)**: not possible with token slices, see
  finding 4. A smaller freeze needs layer-range prefill.
- **`MYNAH_PREFILL_STEP_MS` sweeps on 24L**: inert, see finding 3.

## Next session

Baseline to beat (frozen 2026-09-26): C64, int8 + seg50/first24 + slice 16,
16x2 mb8 -- see "Where we stand". Every experiment must beat it without
breaking it.

1. **Close C64 to GOOD.** Two preferred gates miss: TTFB p95 102.8 ms (the
   16x8 = 128-slot admission is not full at C64, so this is queueing inside a
   worker, not a slot shortage -- check), and 20 of 28328 stalls@250. Profile
   first (`MYNAH_SERVE_PROFILE`, `MYNAH_COST_MAP`): split step time into AR
   step, first-segment prefill, continuation prefill and codec at C64.
2. **Where the ~75-83 audio-s/s ceiling goes.** Throughput is flat C64 -> C72
   (75.0 / 74.3 in 3-minute screens). Candidates, in order: cross-request
   batching of prefill (read the 24L weights once for several prefills),
   prefill rows riding in the decode step's matmul, a DRAM-bandwidth check
   across 16 workers.
3. **Segment-transition bubble** (finding 8) and deadline-aware prefill order
   (streams near underrun > first segment > continuation), only if the
   profile says they are what stalls above C64.
4. Then C72/C80 screens; 30-minute soak before any promotion.
5. Productisation: segmentation is env-gated and off by default. Decide the
   default (upstream semantics say ON for Pocket), add a unit test for
   `src/text_segment.c` and a driver test for `reprepare`, and a 24L serving
   profile under `configs/perf/`.

Open points kept: F32 source pack (`--dtype source`) vs a bf16 pack for load
time/RSS; a per-slice wall-time counter in `MYNAH_SERVE_PROFILE`; layer-range
prefill if a sub-tile freeze is ever needed.

## Reproduce

### Restart recipe (VM recreated, `/dev/shm` empty)

The box's root disk was 97% full, so everything lived in tmpfs `/dev/shm/p24`.

```bash
# on the laptop: ship a clean tree of the commit under test (no .git, small)
git worktree add --detach /tmp/wt-24l <commit>
rsync -a --exclude .git /tmp/wt-24l/ axion:/dev/shm/p24/src/

# on the box: download the 24L checkpoint directly from HF (do not upload it
# from the laptop: 1.3 GB over a slow link). HF token in $HF_TOKEN.
R=492522650173a0653b7575cdc25ae09810e5d741
D=/dev/shm/p24/hf/$R/languages/english_2026-04_24l; mkdir -p $D/embeddings
for f in model.safetensors tokenizer.model embeddings/alba.safetensors; do
  curl -sfL -H "Authorization: Bearer $HF_TOKEN" -o $D/$f \
    https://huggingface.co/kyutai/pocket-tts/resolve/$R/languages/english_2026-04_24l/$f
done
# expected sha256: model.safetensors 8fecefe7206ccfb02fe61a1bd1496d3238f534056f19e5dc0421049cbfbfa4df
#                  alba.safetensors  ff1562ba5293c7f7357220effdc1a9fc97e9f96ebaf721100f86970e0e24c644
cd /dev/shm/p24/src
python3 tools/convert_pocket.py --language english_2026-04_24l \
  --source /dev/shm/p24/hf/$R --output /dev/shm/p24/pocket-en-24l --dtype source
make -j && make server
```

The pack has 358/358 tensors, 24 layers, one voice (`alba`); `numpy` is the
only converter dependency.

### Baseline command (C64, 30 min; needs branch `pocket-24l-cpu-segments`)

```bash
cd /dev/shm/p24/src
unset MYNAH_PREFILL_STEP_MS
MYNAH_QUANT_GROUPS="codec_transformer:int8,codec_conv:int8,codec_convtr:int8,backbone:int8,flow_net:f16,conditioner:f16" \
MYNAH_PREFILL_SLICE=16 MYNAH_POCKET_SEGMENT_TOKENS=50 MYNAH_POCKET_FIRST_SEGMENT_TOKENS=24 \
MYNAH_SERVE_PROFILE=1 python3 tools/serving_profile.py \
  --mode soak --model /dev/shm/p24/pocket-en-24l --bank tests/load_texts_en_v2.txt --language en \
  --levels 64 --soak-seconds 1800 --warmup-seconds 40 --window-seconds 180 \
  --server-bin build/cpu/mynah-tts-server --port 8966 \
  --server-args "--prefork 16 --prefork-threads 2 --max-batch 8"
```

The ASR quality gate is `scratchpad/quality_gate.py` of the 2026-09-26 session
(copied into `reports/20260926-24l-axion/` next to its results); it needs
`~/mynah-asr-v7` and the nemotron model on the box.

`--soak-seconds` must hold at least three `--window-seconds` windows or the
tool refuses to run. For the 10-minute screens use `--soak-seconds 600
--warmup-seconds 40 --window-seconds 60`.

### Operational traps hit this session

- Check the box is idle first: an ASR job from another session was running at
  the start, and a soak beside it measures the other workload.
- The "box is quiet" guard must match processes by exact name
  (`pgrep -x mynah-asr`, `pgrep -x mynah-tts-serve`): `pgrep -f` matches your
  own ssh command lines and waits forever, and `pkill -f <script name>` kills
  your own ssh session.
- **`/dev/shm` is wiped when the user's last session closes** (systemd-logind
  `RemoveIPC`): on 2026-09-26 the pack and build vanished between two ssh
  calls. Keep a session alive for the whole run: `tmux new -d -s keep "sleep
  infinity"`. Copy evidence to disk (`~/p24-evidence`) before stopping.
- Never rsync a tree that was BUILT ON THE MAC: `third_party/ingot/libingot.a`
  is a Mach-O archive and the Linux link fails with undefined `ingot_st_*`.
  Exclude `build`, `*.a`, `*.o`.
- Client coalescing is 16-18% at C48 with 16x2 (35% at 32x1): the load
  generator shares the 32 cores. The tool labels those runs NOT QUOTABLE for
  cadence percentiles; stall counts and throughput are still read, as upper
  bounds on server lateness.

## Measurements

Evidence: `reports/20260925-24l-axion/` (gitignored, local to the laptop):
the client reports of every run below. Per-arm raw client logs and server logs
(`MYNAH_SERVE_PROFILE` step-cost tables) lived in `/dev/shm/p24` and are **lost**
with the VM; the summaries kept the step-cost means.

| file | run |
|---|---|
| `01-c64-c80-16x2-10min.log` | C64/C80, bf16, slice 32, 10 min |
| `02-c48-4x8-mb16-10min.log` | C48, 4x8 mb16, 10 min |
| `03-c48-16x2-10min.log` | C48, bf16, slice 32, 10 min |
| `04-prefill-slice-cap-ab-2min.txt` | slice/cap A/B, 2 min each |
| `05-slice-curve-32x1-cli-int8-3min.txt` | slice 32/24/16/8, 32x1, CLI bf16 vs int8 |
| `06-backbone-bf16-int8-serving-3min.txt` | serving bf16 vs int8, C48/C56 |

### 1. Capacity screens, bf16, slice 32 (shipped defaults), 10 min each

| | 16x2 C48 | 16x2 C64 | 16x2 C80 | 4x8 mb16 C48 |
|---|---|---|---|---|
| verdict | NOT QUOTABLE (coal 17.4%) | NOT QUOTABLE (coal 16.2%) | NOT STREAMABLE | NOT STREAMABLE |
| completed | 7779/7779 | 8391/8391 | 8243/8243 | 3879/3879 |
| TTFB p95 | 48 ms | 106 | 157 | 155 |
| TTFA p95 | 670 ms | 734 | 791 | 957 |
| STREAM_RTF p50/p95 | 0.644/0.746 | 0.799/0.974 | 1.026/1.230 | 1.337/1.559 |
| prebuffer p95 | 74 ms | 373 | 1275 | 7923 |
| stall@250 | 133 (1.7%) | 707 (8.4%) | 28% | 98% |
| stall@500 | 18 (0.2%) | 241 (2.9%) | 19% | 71% |
| audio-s/s | 63.9 | 69.0 | 67.0 | 31.5 |

C48 TTFA p95 by class: short 152 ms, medium 219, conversational 256, long 1102,
italian 1125 (Italian texts on an English-only pack are noise). Drift gates
passed on every level. 4x8 is confounded with `--max-batch 16`, but the gap is
too large for the batch width alone.

### 2. Prefill slice/cap A/B, bf16, C48 16x2, 2 min each (~1530 requests)

| slice / cap ms | TTFA p95 | long p95 | max_gap p95 | stall@250 | stall@500 |
|---|---|---|---|---|---|
| 32 / 40 (default) | 649 | 1154 | 209 | 2% | 6 |
| 32 / 20 | 652 | 1174 | 210 | 2% | 3 |
| 32 / 30 | 644 | 1116 | 209 | 2% | 4 |
| 32 / 60 | 647 | 1058 | 209 | 2% | 2 |
| 16 / 40 | 811 | 1419 | 134 | 1% | 0 |
| 64 / 40 | 604 | 1082 | 354 | 3% | 5 |
| 32 / 40 (repeat) | 648 | 1130 | 210 | 2% | 4 |

STREAM_RTF p95 0.741-0.774 in every arm.

### 3. Slice curve and topology, bf16, C48, 3 min each (~2230 requests)

| arm | TTFA p95 | long p95 | max_gap p95 | stall@250 | stall@500 | audio-s/s |
|---|---|---|---|---|---|---|
| slice 32, 16x2 | 672 | 1188 | 211 | 2% | 9 | 58.1 |
| slice 24, 16x2 | 829 | 1312 | 134 | 1% | 0 | 57.3 |
| slice 16, 16x2 | 829 | 1369 | 134 | 1% | 0 | 57.7 |
| slice 8, 16x2 | 828 | 1397 | 134 | 1% | 0 | 57.3 |
| slice 32, 32x1 | 923 | 1478 | 303 | 2% | 17 | 55.7 (coal 35%) |

Serving step cost at C48, 16x2, bf16: B1 27 ms, B2 40.5, B3 49.7 (mode), B4 55.

### 4. Backbone bf16 vs int8, CLI, 2 threads pinned to 2 cores

`step.backbone` per step, same text and seed, `MYNAH_COST_MAP=2`:

| B | bf16 | int8 | delta |
|---|---|---|---|
| 1 | 21.8 ms | 13.4 | -39% |
| 2 | 26.3 | 21.0 | -20% |
| 4 | 33.9 | 34.2 | 0% |
| 8 | 53.1 | 54.7 | +3% |

Prefill projections unchanged (143 vs 139 ms).

### 5. Backbone bf16 vs int8 in serving, slice 24, 16x2 mb8, 3 min each

| arm | TTFA p95 | long p95 | RTF p95 | gap p95 | stall@250 | stall@500 | audio-s/s | step B3 |
|---|---|---|---|---|---|---|---|---|
| bf16 C48 | 830 | 1375 | 0.750 | 134 | 30 (1.3%) | 0 | 57.4 | 49.8 ms |
| **int8 C48** | **752** | **1253** | **0.668** | **126** | **11 (0.4%)** | **0** | **66.5** | **44.0** |
| int8 C56 | 829 | 1395 | 0.862 | 141 | 64 (2.5%) | 8 | 66.3 | 44.4 |
| bf16 C56 | 849 | 1674 | 0.917 | 148 | 84 (3.7%) | 17 | 59.0 | 51.1 |

Serving B1 step: bf16 27.5 ms, int8 16.7 ms.

### 6. Segmentation A/B, int8, slice 16, C48 16x2 mb8, 3 min each (2026-09-26)

| arm | TTFA p95 | long p95 | medium p95 | RTF p95 | stall@250 | stall@500 | max_gap p95 | audio-s/s | step B3 |
|---|---|---|---|---|---|---|---|---|---|
| no segments | 743 | 1224 | 250 | 0.653 | 9 | 0 | 124 | 67.1 | 42.9 |
| seg 50 | 330 | 463 | 268 | 0.608 | 0 | 0 | 351 | 72.9 | 39.5 |
| seg 50, first <=24 | **275** | **379** | 249 | **0.595** | **0** | 0 | 355 | **73.7** | 39.2 |

### 7. Ladder, int8 + seg50/first24 + slice 16, 16x2 mb8, 3 min each

| C | TTFA p95 | long p95 | TTFB p95 | RTF p95 | stall@250 | stall@500 | max_gap p95 | audio-s/s |
|---|---|---|---|---|---|---|---|---|
| 48 | 275 | 379 | 49 | 0.595 | 0 | 0 | 355 | 73.7 |
| 56 | 311 | 387 | 80 | 0.768 | 4 | 0 | 386 | 72.5 |
| 64 | 340 | 402 | 99 | 0.779 | 1 | 0 | 403 | 75.0 |
| 72 | 378 | 440 | 108 | 0.929 | 4 | **1** | 425 | 74.3 |

C72 fails the mandatory stall@500 (1 of 2844) and TTFB/RTF preferred gates.

### 8. C64, 30 minutes (the baseline)

28328/28328 completed; ten 180 s windows: STREAM_RTF p95 0.773-0.787, TTFA p95
334-348 ms, prebuffer p95 39-43 ms, stall@250 0% per window. Drift PASS (RTF
+0.009, prebuffer +2 ms). stall@500 **0**, stall@250 20 (0.07%), TTFB p95
102.8 ms, max_gap p95 404 / max 795 ms, throughput 82.8 audio-s/s. Step
histogram: B4 341082 frames at 51.3 ms, B3 37686 at 40.9 ms, mean_live 3.88.
Server: 0 failed, 0 rejected, 0 timed out.

### 9. C64 cost model (2026-09-26, baseline config, 5 min, MYNAH_SERVE_PROFILE + MYNAH_COST_MAP=2)

Throughput 74.9 audio-s/s, 0 stalls@500, 1 stall@250 of 4720 -- the same as
the uninstrumented 3-minute screen (75.0), so the instrumentation does not move
the result. (The 30-minute soak reported 82.8; the short screens consistently
read ~75. Unexplained -- likely warm-up/tail accounting in the client over a
short window; do not compare throughput across different soak lengths.)

**Where each worker's wall goes** (16 workers, loop 341.7 s each; the new
`[SERVE] loop` line added on branch `pocket-24l-cpu-segments`):

| component | share of worker wall | source |
|---|---|---|
| AR backbone step | **49.8%** (170.3 s) | `step.backbone` |
| codec | **34.7%** (118.5 s): transformer 17.2%, conv stack 17.4% | `codec.*` |
| prefill | **12.6%**: first segment 8.4%, continuation 4.1% (projections 8.2%, rest 4.4%) | `[SERVE] loop`, `prep.*` |
| flow head + emit | 1.8% | `flow.head`, `step.emit` |
| admission, retire, idle | ~1% (blocked on arrivals 0.8-2.5%) | `[SERVE] loop` |

Coverage ~99%. **No worker imbalance:** step share 85.1-87.0% and prefill
11.6-12.9% across all 16 workers. Prefill slices: first segment 32-33 ms per
16-token slice (~2 ms/token with int8), continuation 46-47 ms per slice
(unexplained; 4% of wall).

**Batch-efficiency map** (CLI, same long text segmented, int8, 2 threads on 2
cores, `--batch B`):

| B | backbone ms/step | backbone ms/slot | codec ms/slot-frame (tf + conv) | flow ms/step |
|---|---|---|---|---|
| 1 | 12.6 | 12.6 | 5.02 (2.51 + 2.48) | 0.69 |
| 2 | 18.9 | 9.4 | 4.98 | 0.87 |
| 3 | 24.4 | 8.1 | 5.01 | 1.01 |
| 4 | 31.3 | 7.8 | 5.08 | 1.09 |
| 5 | 35.8 | 7.2 | 4.98 | 1.41 |
| 6 | 41.6 | 6.9 | 4.96 | 1.56 |
| 8 | 51.9 | 6.5 | 4.95 | 1.75 |

`backbone ≈ 7.4 + 5.6·B` ms; the codec is **flat at ~5 ms per slot-frame**
(decoded per context, no batching benefit). Per slot-frame at B4:
7.8 + 5.1 + 0.3 = 13.2 ms, matching the serving step (13.0 ms/slot). A worker
holds realtime up to `7.4 + 10.8·B <= 80` ms -> **B ≈ 6.7 without prefill,
≈ 5.8 with the measured 12.6% prefill** -> 16 workers x ~4.5-5.8 = the observed
C64-C72 edge. Wider batches or fewer/wider workers cannot help much: the fixed
per-step term is small and the codec does not batch.

**Thread scaling 1 -> 2 threads (CLI, B4):** backbone 44.7 -> 29.5 ms/step
(1.52x), codec transformer 3.30 -> 2.50 ms/frame (**1.32x**), codec conv
3.76 -> 2.44 (1.54x), prefill projections 1469 -> 765 ms (1.92x).

**Reading.** The ceiling is per-slot compute: backbone ~5.6 ms/slot-step and
codec ~5 ms/slot-frame, together ~85% of the wall. At one thread the backbone
does 4 rows x 604 MFLOP in 44.7 ms (~54 GOP/s) while streaming its 302 MB of
int8 weights at ~6.8 GB/s -- far from both the SMMLA compute roof and one
core's bandwidth, so the small-M int8 GEMM looks kernel-limited, not
roof-limited (to be verified with a kernel microbench). Serving at 16 workers
is not slower per step than the isolated CLI, so DRAM contention is not the
limit at C64.

Consequences for the ranked list: prefill batching can win at most ~8% (the
projections' share); batch geometry/topology is closed by the table above.
The two levers left are **(1) the int8 small-M backbone GEMM** (half the wall)
and **(2) the codec per slot** (a third of the wall; its transformer scales
only 1.32x on two threads and never batches across slots).

## 2026-09-26 afternoon: optimisation loop on the C64 cost model

Control for every row: 24L int8 + seg50/first24 + slice 16, 16x2 mb8, 3-minute
screens (180 s + 20 s warm-up), control and candidate back to back.

| ID | hypothesis | change | microbench (CLI, 2 thr) | C64 | C72 | C80 | verdict |
|---|---|---|---|---|---|---|---|
| K1 | the batched int8 path on I8MM hosts reads the weights once per TWO activations (`matvec_q8_pair_i8mm`); the SDOT x4 kernel reads them once per four | none: `MYNAH_QMAT_I8MM=0` | backbone B4 29.2 -> 25.0 ms, B8 50.3 -> 42.8; codec tf 2.45 -> 2.06 ms/frame; prefill proj -41%; WAV byte-identical | TTFA p95 344 -> 275, TTFB p95 105.6 -> 81.6, RTF p95 0.779 -> 0.658, stalls 1/1 -> 0/0, 74.5 -> 88.2 audio-s/s | RTF p95 0.925 -> 0.796, stalls 4/1 -> 0/0, 74.3 -> 88.3 | 0 stalls, RTF 0.801, 88.6 | **WIN** (superseded by K2) |
| K2 | a SMMLA kernel that serves up to 8 activations per weight pass keeps SMMLA's 32 MAC/instr AND the single pass | `matvec_q8_i8mm_np` (x4/x6/x8) in `src/qmat.c`, 4 weight rows x 2*NP activations, 128-bit loads + 64-bit zip, shared epilogue; `MYNAH_QMAT_I8MM_WIDE=0` restores the pair kernel | backbone B4 24.7 (= SDOT), B5 26.5 (SDOT 29.5), B8 35.7 (SDOT 40.4); prefill proj 351 ms (SDOT 444, pair 776); codec tf 2.0; qmat self-test with real I8MM widths 1-9 PASS; WAV byte-identical to pair and SDOT | -- | vs SDOT: RTF p95 0.795 -> 0.753, TTFB 86.5 -> 79.8, 88.3 -> 90.3 audio-s/s, 0/0 stalls | vs SDOT: RTF 0.801 -> 0.766, TTFA 326 -> 308, 88.6 -> 93.0, 0/0 stalls | **WIN, kept** |
| K4 | attention runs on the calling thread only: `perf` (with the env passed through `sudo`!) showed `mynah_dot_f32` + `mynah_axpy_f32` = 22% of a B5 synthesis while the second pool thread waited in `pf_wait_for_work`; codec tf scaled 1.18x from 1 to 2 threads | `tar_attend_head` (one (row, head), shared by both paths) + `tar_attend_rows` dispatching (row, head) pairs to the pool, scores on the task's stack, serial below 64K MACs or a window > 4096; `MYNAH_TAR_ATTN_PARALLEL=0` restores serial | backbone B4 24.7 -> 18.6 ms, B5 26.5 -> 20.0, B8 35.7 -> 27.7; codec tf 2.0 -> 1.45 ms/frame; WAV byte-identical at B1/B4/B5/B8; default config byte-identical to `bea336c` on 6L and 24L | -- | -- | vs K2: TTFA p95 309 -> 249, RTF p95 0.772 -> 0.657, 90.9 -> 106.4 audio-s/s, verdict **GOOD** (coalescing 19% -> 11%: the server now leaves the client room); C96: RTF 0.902 -> 0.762, stall@250 5 -> 0, 91.0 -> 112.5, **GOOD** | **WIN, kept** |
| K5 | with attention on the pool, its inner loop is latency-bound: one accumulator chain of 16 multiply-adds per key, and the output re-loaded/stored per key | NEON head_dim-64 kernels: four keys per pass with one accumulator each (same `vmlaq_f32` order, bit-identical), output held in 16 registers across keys | codec tf 1.45 -> 1.30 ms/frame (-11%), backbone unchanged (B5 20.4 vs 21.5, noise); WAV byte-identical | -- | -- | C112 ABBA: RTF p95 0.872/0.879 (ctl) vs 0.896/0.896 (K5), 112.8/112.1 vs 110.1/111.1 audio-s/s | **REJECTED, removed**: the microbench win does not survive serving and RTF p95 is consistently worse |
| K3 | the int8 conv stack's `mynah_qmat_dots_i8` (SDOT x4 columns) would gain from the same wide SMMLA core | `dots_i8mm_x8`, `CQ8_MAX_BATCH` 4 -> 8 | codec conv 2.47 -> 2.45 ms/frame (noise), WAV identical | -- | -- | -- | **NO EFFECT, reverted.** The int8 conv IS active (forcing it off doubles conv_stack 184 -> 371 ms) but its cost is not in the integer dots |

Notes:
- K1/K2 are exact: every (row, activation) dot is an int32 sum of the same
  products and the float epilogue is the shared `qmat_row_scale`/
  `qmat_row_epilogue`, so audio is byte-identical across pair/SDOT/wide
  (checked by SHA at B1, B4, B8).
- The Mac has no FEAT_I8MM: the wide kernel is only exercised on Axion.
  `self_test_i8mm_ab` now runs widths up to 9 (one x8 group plus a single).

### K2 ladder and qualification

3-minute screens with K2 (wide SMMLA kernel), everything else as the control:

| C | TTFA p95 | TTFB p95 | RTF p95 | stall@250 | stall@500 | audio-s/s |
|---|---|---|---|---|---|---|
| 72 | 299 | 79.8 | 0.753 | 0 | 0 | 90.3 |
| 80 | 308-312 | 86 | 0.766-0.773 | 0 | 0 | 93-94 |
| 88 | 339 | 88.1 | 0.878 | 2 | 0 | 91.3 |
| 96 | 354 | 94.2 | 0.887 | 3 | 0 | 93.2 |

**C80, 30 minutes, K2 -- the new 24L baseline** (2026-09-26 14:23 UTC,
`reports/20260926-24l-axion/soak80wide-30min-20260926-1422.log`):
35175/35175 completed; stall@500 **0**; stall@250 **1** (0.003%); TTFB p95
86.3 ms; TTFA p95 312 ms (long 339, medium 319, conversational 270, short
180); STREAM_RTF p95 0.771, ten windows 0.765-0.777, drift +0.0035;
prebuffer p95 17 ms; max_gap p95 336 ms; throughput **102.9 audio-s/s**; steps
B5 275250 at 52.7 ms (mode), B4 40434 at 45.0 ms. Every gate passes except one
stall@250 in 35175; cadence NOT QUOTABLE (client coalesced 18.2%).

24L capacity on this host: ~C48 (2026-09-25) -> C64 (segmentation) -> **C80**
(K2), i.e. ~0.66x the 6L's 155 audio-s/s.

C80 cost model with K2 (5 min, cost map): backbone 49.6% (at B5), codec 39.2%
(transformer 17.2%, conv stack 21.7%), prefill 10.4% (projections 4.8%), flow
2.6%. SEANet conv stack per frame (`MYNAH_SEANET_PROFILE`): convtr.q8 30.5%,
conv.q8 18.9%, conv.taps (f32) 17.3%, elu 13.7% (already NEON-vectorised),
conv.gemm (f32) 7.6%, rest ~12%.

### K2 + K4 ladder and qualification

3-minute screens: C96 GOOD (TTFA p95 289, RTF 0.762, 0/0 stalls, 112.5
audio-s/s); C104 MARGINAL (RTF 0.857, TTFA 304, stall@250 1, 110.9); C112
MARGINAL (RTF 0.872, TTFA 326, stall@250 1, 113.2); C120 NOT STREAMABLE (RTF
0.951, stall@500 1, stall@250 7, 113.9). Throughput plateau ~114 audio-s/s.

**C96, 30 minutes, K2 + K4 -- QUALIFIED, verdict GOOD** (2026-09-26 15:33 UTC,
`reports/20260926-24l-axion/soak96attn-30min-*.log`): 43113/43113; stall@500
**0**, stall@250 **0**; TTFB p95 81.4 ms; TTFA p95 282 ms (long 292, medium
296, conversational 278, short 157); STREAM_RTF p95 0.763, ten windows
0.759-0.769, drift +0.0024; prebuffer p95 15 ms; max_gap p95 268 ms;
throughput 126.0 audio-s/s; client coalescing 10.4% -> QUOTABLE (warn).

24L on this host: ~C48 (2026-09-25) -> C64 (segmentation) -> C80 (K2) ->
**C96 qualified GOOD** (K4). The 6L small is qualified at C120.

### C104 / C112 tail check (2026-09-26 16:05 UTC)

The ladder's single stall@250 at C104 and at C112 did not reproduce: re-run
back to back, both levels are **GOOD with 0/0 stalls** (4321 and 4364
requests; RTF p95 0.857 / 0.871, TTFA p95 310 / 321). In the K5 ABBA the
control at C112 showed 1 and 3 stalls@250 in ~4350: C112 sits right on the
stall@250 edge, a rare-event level (<= ~1 in 1500), with RTF p95 0.87-0.88.
The load tool's `--json --marks` does NOT carry per-request chunk marks in soak
mode (only summaries and windows), so a single stall cannot be attributed to a
segment transition or a prefill from the client side today; extending the tool
to dump per-request records is the prerequisite for that diagnosis.

### Codec conv campaign: analysed, stopped by the agreed criterion

Stop rule agreed with the owner: 2-3 profiled conv candidates; continue only
with >= ~3% real serving gain, otherwise freeze the 24L baseline.

Conv stack per frame (K2+K4 build, 2 threads, 78 frames, `MYNAH_CONVQ8_PROFILE`
+ `MYNAH_SEANET_PROFILE`; note the int8 rows are numeric, do not grep for
words):

| op | shape m x n x k x taps | ms/frame | GMAC/s |
|---|---|---|---|
| first conv int8 | 512 x 16 x 512 x 7 | 0.17 | ~171 |
| convtr stage 1 int8 | 3072 x 16 x 512 x 1 | 0.15 | ~169 |
| convtr stage 2 int8 | 1280 x 96 x 256 x 1 | 0.22 | ~145 |
| convtr stage 3 int8 | 512 x 480 x 128 x 1 | 0.31 | ~103 |
| 32-channel residual conv, f32 | 32 x 1920 x 64 x 3 | 0.46 | ~26 |
| ELU, 10 passes | ~530 K elements | 0.36 | -- |

Int8 quantisation of the window is 8.7% of the int8 time. Candidates:
1. int8 conv/convtr GEMMs already at ~30-45% of SDOT peak and K3 showed a
   wider kernel does not move them: no low-risk lever.
2. ELU fusion into the following conv: upper bound ~3.4% of the wall if ELU
   vanished entirely; touches carried frame tails and the residual branch.
3. The 32-channel stage runs f32 ON PURPOSE: admitting it to int8 moved
   log-mel correlation 0.9976 -> 0.9785 (`src/convq8.c`, `m < 64` refusal).
   A faster byte-identical f32 kernel for that shape is worth ~2% at most.

No candidate credibly reaches 3% real serving gain at acceptable risk (K5
showed a ~2% local win evaporating in serving), so the campaign stops here.
**Frozen 24L CPU baseline: C96 qualified GOOD** (K2 + K4 + segmentation,
branch `pocket-24l-cpu-segments` at `05b5c03`); C104-C112 is a rare-stall
edge with RTF p95 ~0.86-0.88.

### 6L regression check of the branch (2026-09-26 16:47 UTC)

Shipped default (nothing exported, segmentation off), 6L `models/pocket-en`,
C120, 16x2 mb8, 3 minutes, ABBA `bea336c` vs branch `pocket-24l-cpu-segments`
(`reports/20260926-24l-axion/small-*`):

| | bea336c | branch (K2 + K4) |
|---|---|---|
| verdict | GOOD, GOOD | GOOD, GOOD |
| TTFA p95 | 321, 326 ms | **244, 245 ms** |
| TTFB p95 | 76.0, 76.0 ms | 65.5, 63.9 ms |
| RTF p95 | 0.809, 0.813 | **0.671, 0.667** |
| stall@250 / @500 | 0/0, 0/0 | 0/0, 0/0 |
| audio-s/s | 139.5, 138.9 | **170.3, 171.1 (+22%)** |

No regression: WAV byte-identical to `bea336c` in the default configuration, and
the runtime-general changes (K4 pooled attention on the bf16 backbone and the
codec, K2 on the int8 codec transformer) lift the 6L too. The 6L ceiling above
C120 on this branch is unmeasured; `configs/perf/axion-c4a-32c-pocket-en.json`
still describes `bea336c` numbers.
