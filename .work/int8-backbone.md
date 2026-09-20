# E12 — `backbone:bf16` → `backbone:int8`, measured properly

Status: **OPEN** — written 2026-09-20, before the work. Supersedes E11-10.
Prereq evidence: [`ternary-feasibility.md`](ternary-feasibility.md) §6,
[`backbone-bandwidth.md`](backbone-bandwidth.md),
[`bf16-native-weights.md`](bf16-native-weights.md).

## The claim to test

E11 measured, in Python against the reference implementation, that int8 on the
backbone costs **1.1e-4 mean output NMSE** (6.4e-4 worst) and passes end to end:
**0/7 termination failures at both temperature 0 and the shipped 0.3**, level
correct, duration ratio inside the FP-vs-FP spread of 0.90–1.12.

`backbone-bandwidth.md` measured `step.backbone` at **48.7% of `request.total`**
with eight cores buying 15%, and named the only lever: fewer weight bytes. Per
AR step the backbone reads every weight once — **151.0 MB at bf16, 75.7 MB at
int8**.

So the hypothesis is: halve the traffic on half the wall clock, with kernels
that already ship.

## Two reasons to expect the win to be smaller than that arithmetic suggests

Writing these down *before* the run, because an experiment that cannot come out
against the hypothesis is not an experiment.

**1. At the operating point the weight read is already amortised.** The measured
step model is `T_frame(B) = a + b·B`: **3.0 + 7.8·B ms at f16**, **2.9 + 6.8·B ms
at bf16**. The weight pass happens once per step regardless of `B`, so it lives
in `a`. At the C120 operating point `B` reaches 7–8, where `a = 2.9 ms` is about
**5%** of a 57 ms step. Halving the bytes can only attack that 5%.

The same model says what `b` is: going f16 → bf16 moved **`b` by −13% and left
`a` alone**, and both encodings are two bytes. That was a *kernel* change
(BFMMLA's 16 MACs per instruction against the f16 path's 4), not a bandwidth
change. `b` is compute; `a` is the fixed weight pass.

**2. On Neoverse-V2, int8 and bf16 have the same MACs per instruction.** SMMLA
is 16, BFMMLA is 16. So int8 buys **no arithmetic** over the bf16 we ship today;
it buys bytes, and bytes are what `a` is made of.

**Prediction, falsifiable**: int8 backbone is a **TTFA and low-concurrency
latency** win, not a throughput win. It should move `a` and leave `b` roughly
alone, which means a large effect at B=1 (prefill, first frames, a lightly
loaded box) and a small one at C110–C120. If the soak shows throughput up at the
operating point, this reasoning is wrong and the reason matters more than the
result.

## Why this cannot be measured on the Mac, demonstrated

The A/B needs **no C changes** — `MYNAH_QUANT_GROUPS` already carries it — and a
wiring check on this M1 ran both sides to identical-length audio. It also came
out **23% slower on int8** (RTF 0.245 vs 0.199 at B=1), which is worth exactly
nothing as a signal, because `--dispatch-map` says:

```
isa.arm.i8mm   supported=no   FEAT_I8MM absent   -> batched linear stays on SDOT
isa.arm.bf16   supported=no   FEAT_BF16 absent   -> bf16 weight does not take the BFDOT path
```

Apple M1 is ARMv8.4 and has **neither** of the two units the question is about.
The comparison here is a degraded f16 path against SDOT-only int8; on Axion it
is BFMMLA against SMMLA. **Any number from this host is about this host.**

## The experiment

Target: the GCP Axion `c4a-highcpu-32` (Neoverse-V2), `BLAS=none`, clean rebuild,
tmux, shipped default vs one changed group. Nothing exported by hand — the two
arms are perf-profile entries so the configuration cannot live in shell history.

Arm A (shipped): `codec_transformer:int8,codec_conv:int8,codec_convtr:int8,backbone:bf16,flow_net:f16,conditioner:f16`
Arm B (candidate): the same string with `backbone:int8`.

`MYNAH_QUANT_GROUPS` replaces the **whole** spec, so arm B must spell out the
codec groups too. Getting that wrong silently changes two variables at once,
which is precisely the failure `configs/perf/` exists to prevent: both arms go in
as profile entries with `configuration: environment-override`, and
`tools/perf_profile.py forbidden-env` guards the rest.

### What to measure, beyond RTF

RTF alone cannot distinguish "moved `a`" from "moved `b`", which is the whole
question.

| quantity | how | why it is on the list |
|---|---|---|
| **`step.backbone` ms/frame** | `MYNAH_COST_MAP=2` | the region the change targets; without it the end-to-end delta has no attribution |
| **`a` and `b`, re-fitted** | `--batch 1,2,4,8` sweep, fit `T_frame(B)` | the decisive measurement: which coefficient moved |
| **effective GB/s** | 151.0 or 75.7 MB ÷ measured step time | turns a time into the bandwidth claim, and falsifies it if the number exceeds the machine |
| **TTFA p50/p95, TTFB p95** | `tools/serving_profile.py` | where the prediction says the win is |
| **prebuffer, stall@250/500, RTF p95** | the E5 gate set | a latency win that costs stalls is not a win |
| **throughput, audio-s/s** | soak | where the prediction says there is no win |
| **CPU utilisation, per worker** | `pidstat` / `/proc` during the soak | a bandwidth-bound region freed by fewer bytes should raise utilisation, not lower it |
| **cache misses** | **`perf stat -e cache-misses,LLC-load-misses`** externally | we have **no in-process perf counters** — `src/costmap.h` is wall-clock regions only. `perf stat` on the box is the whole story here; do not claim a cache effect without it |
| **peak RSS** | existing bench output | int8 weights are a second cached copy; the packer's footprint is not free |

### Gates

Promotion needs **all** of:

1. the E5 mandatory gates at the current operating point — `completed == launched`,
   `STREAM_RTF p95 < 1.00`, `stall_rate@500ms == 0`;
2. TTFA p95 **no worse** than the bf16 arm, and the preferred set intact;
3. a **30-minute soak**, not a 10-minute screen — a screen may not promote;
4. a listening pass on the v2 corpus at the shipped temperature, under the
   supreme rule: if it sounds good it wins, if it does not, nothing else counts;
5. determinism unchanged — same seed, same text, N repetitions under 0/7/23/47
   concurrent load, identical sha256, on a clean rebuild. int8 changes the
   numerics; it must not make them *non-reproducible*.

Any single failure leaves `POCKET_QG_DEFAULT_SPEC` alone and the result goes in
`docs/performance.md` as a measured negative.

## What E11 did and did not license

E11's int8 evidence is a **screen**: Python, on a Mac, against the upstream
implementation, seven sentences, one voice, one language — **not** against
`engine_pocket.c` and not under load. It is enough to justify spending a box on
this. It is not enough to change a default, and the two must not be confused
when this note is read back.
