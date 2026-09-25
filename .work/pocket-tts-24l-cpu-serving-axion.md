# PocketTTS 24L on CPU: streaming capacity on Axion (first screens)

Status: **open, first screens only** · 2026-09-25 · no operating point yet

Host: GCP Axion `c4a-highcpu-32`, 32x Neoverse-V2, 62 GB, Linux, `BLAS=none`,
`SIMD=auto`. Source `010020d` (HEAD of `pocket-cuda-streaming-parity`, clean
worktree without the uncommitted CUDA work). Pack: official English 24L at
revision `492522650173a0653b7575cdc25ae09810e5d741`, converted on the box with
`--dtype source` (358/358 tensors, `model.safetensors` sha256 `8fecefe7…`),
voice `alba` only. Shipped defaults, nothing exported (`MYNAH_QUANT_GROUPS`,
`MYNAH_PREFILL_SLICE`, `MYNAH_PREFILL_STEP_MS` unset). Mixed v2 bank,
`/v1/audio/speech` streaming, load generator on the same host. Every level is a
**10-minute screen**: a screen cannot promote and nothing here is a qualification.

Logs lived in `/dev/shm/p24/` on the box (tmpfs, lost on reboot):
`soak24-*.log`, `soak24-4x8-*.log`, `soak24-16x2c48-*.log`.

## Question

The 6L small pack qualifies at C120/C126 GOOD on this host (16x2, `--max-batch 8`,
~155 audio-s/s). Does the 24L large pack hold a useful concurrency with the same
serving stack, and where does it degrade?

## Measured

Single request, CLI, 2 threads (the per-worker width), same sentence and seed:

| pack | RTF | step.total / step |
|---|---|---|
| 6L | 0.385 | ~11 ms (serving B1) |
| 24L | 0.41 | ~20 ms (73 steps, 1467 ms) |

At C1 the 24L costs almost nothing extra. Under batching it does:

| 24L screen, 10 min | 16x2 mb8 C48 | 16x2 mb8 C64 | 16x2 mb8 C80 | 4x8 mb16 C48 |
|---|---|---|---|---|
| verdict | NOT QUOTABLE (coal 17.4%) | NOT QUOTABLE (coal 16.2%) | NOT STREAMABLE | NOT STREAMABLE |
| completed | 7779/7779 | 8391/8391 | 8243/8243 | 3879/3879 |
| TTFB p95 | 48 ms | 106 | 157 | 155 |
| TTFA p95 | **670 ms** | 734 | 791 | 957 |
| STREAM_RTF p50/p95 | 0.644/**0.746** | 0.799/0.974 | 1.026/1.230 | 1.337/1.559 |
| prebuffer p95 | 74 ms | 373 | 1275 | 7923 |
| stall@250ms | 133 (1.7%) | 707 (8.4%) | 28% | 98% |
| stall@500ms | **18 (0.2%)** | 241 (2.9%) | 19% | 71% |
| throughput audio-s/s | 63.9 | 69.0 | 67.0 | **31.5** |

C48 TTFA p95 by class: short 152 ms, medium 219, conversational 256, **long
1102**, italian 1125 (Italian texts on an English-only pack are noise).

## Readings

1. **Throughput ceiling ~67-69 audio-s/s at 16x2**, about 0.44x the small pack's
   155. C64 -> C80 adds no throughput, only queue: that is the compute roof.
2. **4x8 is decisively worse** (31.5 audio-s/s, less than half). Confounded with
   `--max-batch 16` (required: 4x8 slots would be 32 < C48), but the gap is too
   large for the batch width alone. Same direction as the small pack
   (16x2 beat 8x4 there too), opposite to qwen-tts 1.7B on this host.
3. **At C48 capacity is fine** (RTF p95 0.746, flat over ten windows); what fails
   is TTFA (long texts) and a small stall tail. Short/medium traffic is served
   well.
4. **Hypothesis, not measured:** both failures come from prefill. A 24L prefill
   is ~4x the 6L one, and `MYNAH_PREFILL_STEP_MS=40` was measured for the 6L
   step cost. With a dearer AR step the per-frame slack is smaller, so long-text
   prefills (a) take more steps to finish -> TTFA, and (b) still overrun frames
   of co-resident slots -> stalls. Same mechanism the small pack had on
   2026-09-19, closed then by re-measuring the cap.
5. The instrument: client coalescing 16-17% at C48/C64 (refusal at 15%) — load
   generator on the same 32 cores. Cadence numbers are upper bounds.

## Next (plan, not done)

1. `MYNAH_COST_MAP` / `MYNAH_SERVE_PROFILE=1` at C48 on 24L: read `T_frame(B)`
   (a + b·B) and prefill cost per slice, instead of guessing from the small.
2. Prefill-cap sweep for 24L at C48 (e.g. 20/30/40/60 ms), 10 min each, one
   variable. Watch TTFA p95 of the `long` class and stall@500.
3. C40 screen at the shipped default as the fallback operating point.
4. Only then a 30-minute soak at the best level; a screen may not promote.
5. Consider whether the 24L needs its own serving profile
   (`configs/perf/axion-c4a-32c-pocket-en-24l.json`) rather than inheriting the
   6L defaults.
6. Open: F32 source weights vs a bf16 pack for 24L (`--dtype source` vs `bf16`):
   runtime groups already run `backbone:bf16`, but check load time/RSS and that
   the quant groups actually applied to the F32 tensors.
