# Pocket CUDA: hierarchical KV (VRAM → pinned RAM → local NVMe) — desk analysis before any code

Status: measured 2026-09-30 (see "Measured" below). Probe code behind MYNAH_CUDA_KV_OFFLOAD_POC (off by default). Verdict so far: **NO-GO for capacity on L4 24L**. The limit at
C160+ is step time, not memory. The cheap falsification run below can confirm this. The adjacent ideas at the end
are the better next bets.

## Facts used
- **Box:** AWS g6.xlarge. 1× L4 with 22 GiB usable, 4 vCPU, **15 GiB RAM** (`free -g`), local instance NVMe
  **232.8 GB** (`nvme1n1`, not mounted), and an EBS root volume.
- **24L backbone KV:** d_model 1024 with 24 layers, so K+V per position per row is 2 × 1024 × 24 × dtype bytes:
  | KV dtype | Per position per row | 250 positions | 617 positions (a fixed-capacity design) |
  |---|---|---|---|
  | FP32 | 192 KiB | 47 MiB | 116 MiB |
  | BF16 (current) | 96 KiB | 23 MiB | 58 MiB |
  | INT8 + per-head scales | ~48 KiB | ~12 MiB | ~29 MiB |
- **Positions per row:**
  - Voice prompt: ≤250 at 12.5 Hz for 20 s; typical voices are 9–11 s, i.e. 112–137 positions.
  - Text segment: ≤24 tokens for the first segment, ≤50 after.
  - Generated frames: 1.7 s short = 21, 3.8 s conversational = 48, 6.7 s medium = 84.
  - A typical row is therefore ~200–300 positions, or 19–29 MiB in BF16. KV_GROW allocates in 256-position chunks.
- **Measured VRAM:**
  - This engine, v1.5.0, bf16 KV, KV_GROW: 2.1 GB after start and 15.3 GB at the end of a C160 run, so **~82 MB per
    live row all-in** (KV headroom, codec/decoder state, per-row buffers).
  - A fixed-capacity PyTorch continuous-batching server with bf16 KV and a windowed Mimi ring on the same L4:
    14.4 GB at 160 rows and 16.4 GB at 192 rows.
- **Measured streaming limit,** same L4 (closed loop, gate STREAM_RTF p95 < 0.90):
  - Both engines are GOOD at C160 (RTF p95 0.84–0.85) and MARGINAL at C176 (0.93–0.94).
  - At C192 the PyTorch server has 16.4 GB in use, ~5.6 GB free, and still fails on RTF (0.97).
  - Aggregate speech plateaus at ~170–200 audio-s/s from C128 up.

## Reading
1. **Memory is not binding at the realtime knee.** At C192, memory still has ~5 GB spare and RTF has already failed.
   - Approximate VRAM(C) for this engine: ~2.1 GB static + C × ~0.08 GB. That predicts ~12.6 GB at C128,
     ~17.5 GB at C192 and ~22.6 GB at C256; KV_GROW growth on long rows would push these higher.
   - So C256 would need more memory. **C256 would not be realtime anyway:** at batch 256 the step exceeds the 80 ms
     frame (C176 is already at 0.93 × 80 ms).
2. **Tiering adds logical sessions, not realtime capacity.**
   - An actively speaking stream must be stepped every 80 ms. It is never "cold" while it speaks.
   - Pocket KV is request-scoped. When a request ends its KV is discarded, and a queued request has no KV yet, so no
     parked KV exists to tier.
   - The only way to create parked KV is time-slicing: generate ahead to build a client-side lead, then park the
     stream. That cannot beat the aggregate bound, which is simultaneously realtime streams ≤ aggregate audio-s/s
     ≈ 170–200 on this GPU. Narrower batches also lower that aggregate.
3. **Host RAM and NVMe numbers,** for completeness:
   - ~10 GiB of pinnable RAM on g6.xlarge holds ~350–500 BF16 rows at 20–29 MiB.
   - A 25 MiB restore over PCIe Gen4 x8/x16 is ~2–4 ms.
   - From NVMe, a 25 MiB read is ~12–25 ms at 1–2 GB/s.
   - Latency would therefore be tolerable. **There is no workload that needs it** under request-scoped KV.
4. **When the idea would matter** (not today):
   - multi-turn session state kept across requests (reusing a speaker/prosody context between turns);
   - much longer contexts;
   - a GPU with far less memory per unit of compute;
   - an offline, non-realtime batch mode.

## Cheap falsification run (only if we want the curve on record; ~15 min on the g6)
v1.5.0 profile with bf16 KV and `--max-batch/--max-inflight 256`, closed-loop 60-s levels at C32/64/96/128/160/192/256.
Record `nvidia-smi` used/free, `/health` device_memory_free, the MB/row slope, the `[SERVE]` step mean per width,
audio-s/s, TTFA and RTF p95, stalls, and the failure mode (OOM vs RTF).
- **Expected:** a linear ~80 MB/row, and RTF failing at ~C176 long before OOM.
- **GO for tiering only if:** memory fails before RTF at some C while the step time still has headroom.

## Better next bets (they attack step time, the actual limit)
1. **Shared voice-prefix KV with prefix-shared attention.**
   - Rows using the same voice read the prompt KV once per step per head instead of once per row (cascade /
     shared-prefix attention).
   - The prompt is ~40–60% of a typical row's positions. This cuts attention bytes read and KV memory, and it removes
     the ~48 per-row D2D prompt copies at admission.
   - It fits the per-voice VoicePrompt already cached.
2. **INT8 KV as a bandwidth lever, not a capacity lever.**
   - Attention reads half the bytes of BF16. The ragged/length-aware attention showed that KV bytes read is a
     first-order share of step time.
   - Needs per-head scales, a dequantizing attention kernel and a quality gate (WER, listening, log-spectral
     distance). Never on by default without the gate.
3. The first-chunk items in `.work/pocket-cuda-cold-burst.md` (Next section).

## Decision 2026-09-30: try it anyway, after the burst round
The desk verdict stands (NO-GO for capacity). We still want one cheap attempt after the burst experiments
(E15-29 parts 4–5), using short screens only:
1. The falsification ladder above: C32→C256, 60-s levels, VRAM/MB-per-row/RTF/failure mode (~15 min).
2. The smallest host-RAM PoC: one BF16 slot's KV evicted to pinned RAM and restored while C128 runs.
   - Measure bytes, D2H/H2D latency, effective PCIe bandwidth and overlap with the running batch.
   - Most important: stalls and gaps of the running streams.
   - Only with a flag on the CUDA path.
3. NVMe (`nvme1n1`, 233 GB, unmounted) only if step 2 shows a use: `fio` sequential read/write at 16–64 MiB blocks.

## Queued idea (2026-09-30): 8-bit KV on the CUDA path (after tiering and the long soaks)
The L4 is Ada (sm_89) with hardware FP8 (E4M3) and INT8, so an 8-bit device KV is realistic. The lever is attention
bandwidth, not capacity: half the bytes of BF16 per step, and bytes read are a first-order share of step time. Options:
- INT8 with per-head (or per-channel) scales, stored next to the KV;
- FP8 E4M3 with a per-head scale (simpler dequantization).
QKV projections and attention accumulation stay FP32, as for BF16 KV.

Quick falsification, in this order:
1. Quality: quantize-dequantize on write (flag, the scalar path is fine). Compare to BF16 at fixed seed with the
   oracle-parity tolerances and log-spectral distance, then WER on ~100 utterances.
2. Speed: an attention microbench with 8-bit loads and in-kernel dequantization at B160, T~300.

GO only if quality holds and attention gets ≥ 1.3× faster. Then run a C176/C192 screen.
- Test **both** 8-bit formats side by side in the same harness: INT8 with per-head/per-channel scales, and FP8 E4M3
  with a per-head scale (optionally E5M2 as a range check). The two formats trade off differently:
  - FP8 keeps a floating exponent, so it handles outliers better. Dequantization is one conversion, and Ada supports
    the format natively.
  - INT8 has a uniform grid, so it is more precise inside the range, but it depends on good scales.
  Choose on quality first, then on kernel speed.

## Measured (2026-09-30)
## Step 1: ladder with --max-batch/--max-inflight 256, one fresh server

| C | VRAM used (max) | device free | audio-s/s | TTFA p95 | STREAM_RTF p95 | stalls@250/500 |
|---|---|---|---|---|---|---|
| 32 | 4.9 GB | 18.7 GB | 113.7 | 55 ms | 0.261 | 0/0 |
| 64 | 7.5 GB | 15.9 GB | 137.9 | 80 ms | 0.415 | 0/0 |
| 96 | 10.2 GB | 12.9 GB | 146.5 | 103 ms | 0.570 | 0/0 |
| 128 | 12.8 GB | 10.2 GB | 152.1 | 129 ms | 0.709 | 0/0 |
| 160 | 15.2 GB | 7.8 GB | 155.6 | 153 ms | 0.860 | 0/0 |
| 192 | 17.8 GB | 5.0 GB | 154.7 | 175 ms | **1.005** | 0/0 |
| 224 | 20.4 GB | 2.4 GB | 152.6 | 201 ms | 1.157 | 74k/54k (s) |
| 256 | 22.5 GB | 0.03 GB | 153.9 | 225 ms | 1.307 | 97k/80k (s) |

## Step 2: one-slot evict/restore probe (`MYNAH_CUDA_KV_OFFLOAD_POC=1`) while C128 runs, 90 s per arm

The probe takes the live row with the most positions, copies the used prefix of its KV (48 strided rows) to pinned
RAM on a side stream, then restores it into the row's buffer (later compute waits for the restore), and verifies the
round trip.

| Arm | Rounds | Bytes per slot, p50 / max | D2H p50 / p95 | H2D p50 / p95 | Effective PCIe | STREAM_RTF p95 | audio-s/s | stalls |
|---|---|---|---|---|---|---|---|---|
| off | — | — | — | — | — | 0.713 | 160.2 | 0/0 |
| on, every 2 s | 56 | 32.4 / 40.3 MiB | 2.58 / 2.90 ms | 2.36 / 2.48 ms | 13.2–13.4 GB/s | 0.714 | 159.9 | 0/0 |
| on, every 0.5 s | 183 | 32.3 / 41.1 MiB | 2.58 / 2.95 ms | 2.39 / 2.50 ms | 13.2–13.4 GB/s | 0.724 | 156.8 | 0/0 |

- Verification failed in 2 of 56 and 5 of 183 rounds.
  - Likely cause: a probe-side race, not the transfer. The probe identifies its target by context pointer; if the
    request retires and a new one reuses that context or its pooled KV set between evict and restore, the restore
    writes the old prefix into a different request.
  - A real tier manager would key on a request id and pin the slot. This does not change the transfer-cost facts.
- Throughput cost: none measurable at 1 round every 2 s, and −2 % throughput plus +1.5 % RTF p95 at 2 rounds per
  second.

## The 10 points
1. **VRAM equation (measured, R² ≈ 1):** VRAM(C) ≈ 2.5 GB + 79 MiB × C. The batch-dependent term is not separable at
   this resolution, since graphs grow with width. Linear from C32 to C256.
2. **BF16 KV per row:**
   - From code: 96 KiB per position. The initial capacity is voice + text + max(256, 3×text+64) positions, rounded to
     256-position chunks, i.e. ~400–450 positions or **~38–42 MiB** at admission.
   - Measured: the used prefix at C128 is p50 346 positions, i.e. **32 MiB**.
   - The remaining ~37–40 MiB per row is codec, decoder and staging state, so KV is about half of the per-row
     footprint.
3. **Predicted memory:** C128 12.6 GB, C192 17.7 GB, C256 22.8 GB, which matches the measurements.
4. **Actual failure mode: compute, not memory.**
   - Aggregate speech plateaus at ~153–156 audio-s/s from C128 up.
   - STREAM_RTF p95 crosses 0.90 between C160 and C192, and passes 1.0 at C192 **with 5 GB free**.
   - OOM would only come at ~C256, where streaming has long failed (RTF 1.31, stalls everywhere).
5. **Host-RAM tiering, feasibility:**
   - ~10 GiB of pinnable RAM holds ~300 KV prefixes of 32 MiB.
   - Mechanically it works: side-stream copies overlap compute at 13 GB/s with no stalls.
   - But it only adds logical sessions, which do not exist in this workload, since KV is request-scoped.
6. **Eviction / restore latency:**
   - 2.4–2.9 ms per 32–41 MiB slot, D2H and H2D alike (13.2–13.4 GB/s over PCIe Gen4 on g6).
   - That is ~3 % of one 80 ms frame, so a restore fits well inside any stream's slack (≥250 ms of client lead).
   - Latency is not the problem.
7. **NVMe:** not needed.
   - RAM already holds every state that could exist.
   - At an estimated 1–2 GB/s a 32 MiB read costs ~16–32 ms, which is still fine for a resume.
   - There is no capacity to gain, because memory is not the binding resource. Step 3 was not run.
8. **INT8 KV before tiering?** Not for capacity: memory is not binding at the realtime knee. Possibly as a bandwidth
   lever, since attention KV reads are a first-order share of step time. That needs a dequantising attention kernel and
   a WER/listening gate, and the realistic gain is a fraction of the attention share, not 2×.
9. **Smallest PoC:** done (the probe above). It leaves the transfer-cost facts on record. Anything bigger (a tier
   manager, pinned scheduling) has no workload to serve at this model size on an L4.
10. **Verdict: NO-GO** for KV tiering as a capacity lever for 24L Pocket on L4. GO conditions to revisit:
    - memory fails before STREAM_RTF at some C while step time still has headroom, e.g. a much faster step, a larger
      model, or longer contexts; or
    - KV must persist across requests (multi-turn session reuse), where host-RAM parking at ~2.5 ms per restore is
      clearly viable and NVMe would serve as the cold tier.

The levers for C176+ attack step time: shared voice-prefix KV with prefix-shared attention (measured on a PyTorch
continuous-batching engine: step −10–15 %, C192 GOOD over 10 min), then INT8 KV as a bandwidth lever.
