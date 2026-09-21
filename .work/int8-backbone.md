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

---

# Measured, 2026-09-20, GCP Axion c4a-highcpu-32 (Neoverse-V2)

Host has **both** units the Mac lacked: `isa.arm.i8mm` ON (SMMLA, wired into the
weight-stationary batched linear) and `isa.arm.bf16` ON. `BLAS=none/mynah-sgemm`,
`SIMD=auto (arm64/native)`, clean rebuild at `e7b2a9d`.

## E12-3: the prediction held, and the numbers are cleaner than expected

`MYNAH_COST_MAP=2`, CLI, single thread, `step.backbone` over 77 steps, the same
utterance and seed on both arms:

| B | bf16 ms/step | int8 ms/step | delta |
|---|---|---|---|
| 1 | 2.427 | 2.157 | **−11.1%** |
| 2 | 3.335 | 3.167 | −5.0% |
| 4 | 5.245 | 4.893 | −6.7% |
| 8 | 8.285 | 8.262 | **−0.3%** |

Least-squares over all four points:

```
bf16:  T_backbone(B) = 1.691 + 0.835*B  ms
int8:  T_backbone(B) = 1.377 + 0.865*B  ms
```

**int8 moves `a` by −18.6% and `b` by +3.6%.** That is the prediction recorded in
E12-3 before the run, confirmed on both coefficients and in the right direction
on each: the weight pass is once per step and lives in `a`; the per-slot term is
compute, and int8 does not help it because SMMLA and BFMMLA are both 16 MACs per
instruction. `b` is very slightly *worse*, which is what paying to quantize the
activation once per step looks like.

**Consequence for the operating point.** C120–C130 runs at B ≈ 7–8, where the
two arms are within 0.3% of each other. There is no throughput win there, and
this was knowable from `T_frame(B) = a + b·B` before a single soak was run. The
win is at B=1: prefill, TTFA, a lightly loaded box.

## One thing the prediction got wrong, and it matters

`a` fell by **18.6%, not by half**, while the weight bytes fell by half
(151.0 → 75.7 MB). So `a` is **not purely the weight pass**. Something fixed
lives in there — per-step setup, the activation quantization, the epilogue,
pool dispatch — and a bandwidth model that assumes `a` is all bytes will
over-predict every future weight-shrinking change by roughly 2.7x.

That is worth more than the int8 result itself: it is a correction to the
instrument that `backbone-bandwidth.md` reasons with. The next weight-format
change should be predicted against `a_bytes ≈ 0.31 ms per 75 MB` and a fixed
remainder of ≈ 1.06 ms, not against `a` as a whole.

## Reporting gap found on the way

`--dispatch-map`'s `isa.arm.bf16` row describes **BFDOT** ("eight MACs per
instruction"), but `matvec_bf16_neon_tile` — BFMMLA, sixteen MACs, eight
accumulators — is what `qmat_rows_job` actually calls for the batched path
(`src/qmat.c:3638,3654`). The row is true of the single-vector decode matvec and
understates the batched one. Anyone reading the dispatch map to decide whether
bf16 is competitive at batch would conclude it is half as wide as it is.

## E12 serving-level A/B at C130: int8 is a no-op under load

Same build `e7b2a9d`, same box, 10-minute screens back to back, ~21,700
requests each, 0 failed and 0 rejected on both arms.

| C130, 10 min | int8 | bf16 (shipped) | delta |
|---|---|---|---|
| TTFB p95 | 199.0 ms | 203.5 ms | −2.2% |
| **TTFA p95** | 409.2 ms | **402.8 ms** | **+1.6%, int8 worse** |
| STREAM_RTF p95 | 0.822 | 0.820 | +0.2% |
| prebuffer p95 | 44.3 ms | 43.0 ms | +3% |
| requests | 21,702 | 21,742 | −0.2% |
| stall@250 / @500 | 0% / 0% | 0% / 0% | = |
| verdict | MARGINAL | MARGINAL | = |

Both MARGINAL for the same single reason: **TTFB p95 ≈ 200 ms against the
100 ms preferred gate**. Every mandatory gate passed on both. That is the
128-slot wall — 16 workers × `--max-batch 8` — and it is identical on the two
arms, which is what "the dtype is not the constraint here" looks like.

This is the serving-level confirmation of the batch sweep: at B ≈ 8 the two
arms differ by 0.3% in `step.backbone`, and 0.3% of a region is not visible in
an end-to-end percentile.

**An attribution error I made and had to retract mid-run.** On seeing int8's
TTFA p95 of 409.2 I compared it against yesterday's C130 bf16 number (580.9 ms)
and reported a 29.5% win. Today's bf16 arm returns **402.8 ms**. The improvement
is the *build*, not the dtype. Yesterday's run was a different commit and a
different binary; the only admissible control is the arm measured beside it, on
the same build, on the same box, in the same hour. The control existed
specifically to prevent this and it still nearly went out as a finding.

## Two operational traps, both mine, both worth keeping

**`tmux kill-session` does not kill what the session started.** The C140 screen
outlived its session and ran concurrently with the first soak attempt. A soak
that shared the box with another 16-worker server is not a soak.

**`pkill -x mynah-tts-server` matches nothing and reports success.** Linux
truncates `/proc/<pid>/comm` to 15 characters, so the process name is
`mynah-tts-serve` (15) and an exact match on the 16-character name silently
finds nobody. `pgrep -c -x` then returns 0 and the script prints "servers still
alive: 0" while three are running — `ps -C mynah-tts-server` lists them because
it does not truncate. **Kill by PID from `ps -C`, and prove the box is empty
with a different tool than the one that did the killing.** A cleanup that
self-certifies is not evidence.

## C128, thirty minutes: MARGINAL by three stalls in 65,295

Run on the int8 arm, 20 Sep, `e7b2a9d`, box proven empty before the start.
65,295 requests, 1845 s wall, ten 180-second windows all flat.

```
PASS mandatory  completed == launched    65295 == 65295
PASS mandatory  STREAM_RTF p95           0.818 < 1.000
PASS mandatory  stall_rate@500ms         0 of 65295
PASS preferred  TTFB p95                  77.4 ms  <= 100
PASS preferred  TTFA p95                 336.3 ms  <= 500
PASS preferred  STREAM_RTF p95           0.818 <= 0.900
PASS preferred  required_prebuffer p95    40.3 ms  <= 500
PASS preferred  safe_play_start p95      372.5 ms  <= 1000
FAIL preferred  stall_rate@250ms         3 of 65,295          <- the only miss
PASS drift      prebuffer  last vs best  +0.0083 (tol +0.1500)
PASS drift      stream_rtf last vs best  +0.0000 (tol +0.0500)
```

Throughput **158.2 audio-s per wall-s**. Three stalls is 0.005% of requests. The
gate is written as exactly zero deliberately and the verdict stands, but the
distance from qualifying is three interruptions, not a margin.

## The ceiling is a slot count, and the measurement matches the arithmetic

16 workers x `--max-batch 8` = **128 places**. Time to accept a request:

| level | places | TTFB p95 | verdict |
|---|---|---|---|
| C120 | 8 spare | 75.1 ms | GOOD, 30 min |
| C128 | exactly full | **77.4 ms** | MARGINAL, 30 min, 3 stalls |
| C130 | 2 over | **203.5 ms** | MARGINAL, 10 min |

Flat at 75–77 ms right up to the 128th stream, then nearly triples with two
more. **The next capacity increase is a configuration change — worker count or
`--max-batch` — not a kernel.** Nothing in E12 touched that, and it is now the
cheapest lever on the board.

## E12 verdict: int8 backbone NOT promoted

Gate 2 of this note's own list — "TTFA p95 no worse than the bf16 arm" — is
missed: 409.2 against 402.8 at C130 on the same build. Gates 1 and 3 pass, gates
4 and 5 were never reached because there is no reason to spend a listening pass
and a determinism sweep on a change measured at 0.3% where it would ship.

`POCKET_QG_DEFAULT_SPEC` stays as it is. The negative goes to
`docs/performance.md`.

What int8 *is* good for, unmeasured here: **memory**. 151.0 → 75.7 MB of backbone
weights, and at 16 prefork workers that is worth checking if the quantized cache
is per-process rather than shared. No RSS was captured — the server log does not
emit it — so this is a hypothesis, not a result, and it is the only remaining
reason to revisit int8.

---

# E13: finding the certifiable level, 2026-09-21

Fresh Axion `c4a-highcpu-32`, `136.112.223.214`. Synced and **clean-rebuilt**
before measuring, verified rather than assumed:

```
remote        https://github.com/mynah-org/mynah-tts.git
HEAD          e7b2a9d == origin/main
tracked mods  0
git diff origin/main -- src server cli tools tests Makefile   ->  empty
binary        build/cpu/mynah-tts-server  2026-09-21_13:14   (rebuilt, not reused)
isa.arm.bf16 ON   isa.arm.i8mm ON   BLAS=none/mynah-sgemm   SIMD=auto (arm64/native)
```

The empty diff is stronger evidence than a fresh clone would be: a clone proves
the files *came from* origin, the diff proves the files *are* origin, byte for
byte, on every source path.

## Method: a ladder that walks itself down

One unattended script: screen a level for 10 minutes, and **promote the first
GOOD straight to 30 minutes** rather than round-tripping through a human. Walk
126 -> 124 -> 122 and stop at the first that qualifies. Shipped default
throughout — int8 was measured and rejected on 20 September, and only the
shipped configuration is quotable anyway.

Each level is preceded by a kill-and-prove: PIDs from `ps -C`, then a printed
count of what survived. That is there because on 20 September a screen outlived
its tmux session and competed with a soak for the box.

## C126, 10-minute screen: GOOD on all nine gates

```
PASS mandatory  completed == launched   21641 == 21641
PASS mandatory  STREAM_RTF p95          0.815 < 1.000
PASS mandatory  stall_rate@500ms        0 of 21641
PASS preferred  TTFB p95                 78.8 ms <= 100
PASS preferred  TTFA p95                330.7 ms <= 500
PASS preferred  STREAM_RTF p95          0.815 <= 0.900
PASS preferred  required_prebuffer p95   38.0 ms <= 500
PASS preferred  safe_play_start p95     365.5 ms <= 1000
PASS preferred  stall_rate@250ms        0 of 21641      <- the gate C128 lost
```

Not a squeak-through: TTFB uses 79% of its budget, TTFA 66%, RTF 91%. The
30-minute confirmation was promoted automatically and is the run that may
certify.

## The frontier is narrow, and that is the useful part

| level | soak | requests | stall@250 | verdict |
|---|---|---|---|---|
| C120 | 30 min | 64,205 | 0 | GOOD (certified 19 Sep) |
| C126 | 10 min | 21,641 | **0** | **GOOD** |
| C128 | 30 min | 65,295 | 3 | MARGINAL |
| C130 | 10 min | 21,742 | 0 | MARGINAL (TTFB 203 ms, queue full) |

Between a clean 126 and a 128 that lost on three interruptions in 65,295 there
is only 127 — and the open question of whether C128's three were luck. Note the
two MARGINALs fail for *different* reasons: C128 on stalls, C130 on the queue.

## C126, thirty minutes: GOOD — the certified point moves 120 -> 126

21 Sep, fresh Axion, clean rebuild at `e7b2a9d`, shipped default, box proven
empty before the run.

```
65,039 requests   completed == launched   0 failed
PASS all nine gates
  TTFB p95        79.1 ms   <= 100
  TTFA p95       329.5 ms   <= 500
  STREAM_RTF p95   0.815    <= 0.900
  prebuffer p95   40.7 ms   <= 500
  safe_start p95 367.4 ms   <= 1000
  stall@250ms      0 of 65,039      <- the gate C128 lost
  stall@500ms      0 of 65,039
drift stream_rtf  +0.0000 over ten 180 s windows
throughput       157.4 audio-s per wall-s
```

**+5% on the certified figure, and the frontier is now tight**: 126 clean over
65,039 requests, 128 stopped by 3 stalls over 65,295. Only 127 lies between them.

The ladder that produced this promoted the first GOOD screen to thirty minutes
without a human round trip, which is the shape worth reusing: a screen may not
promote, but it can *choose what to soak*, and that turns a 40-minute question
into one unattended run.

| level | soak | requests | TTFB p95 | TTFA p95 | stall@250 | verdict |
|---|---|---|---|---|---|---|
| C120 | 30 min | 64,205 | 75.1 ms | 447.4 ms | 0 | GOOD (19 Sep build) |
| **C126** | **30 min** | **65,039** | **79.1 ms** | **329.5 ms** | **0** | **GOOD — certified** |
| C128 | 30 min | 65,295 | 77.4 ms | 336.3 ms | 3 | MARGINAL |
| C130 | 10 min | 21,742 | 203.5 ms | 402.8 ms | 0 | MARGINAL (queue) |

Note the two MARGINALs fail for **different** reasons — C128 on stalls, C130 on
the queue — so they are not one wall seen twice.
