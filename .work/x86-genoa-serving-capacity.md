# x86 Genoa serving capacity: what is measured, what is NOT, and the number I got wrong

Board: `PLAN.md` §E14-9. Host: AMD EPYC 9254 (Genoa, Zen 4), 24 cores, Ubuntu
24.04, gcc 13.3, `BLAS=none`. Pack `models/pocket-en`. 2026-09-22.
Box released the same evening; everything below is what survived it.

## THE HEADLINE IS NOT A CAPACITY NUMBER

**C80 is measured for a workload nobody would serve.** It is real, it is
reproducible, and it does not answer "how many streams does this box hold",
because the load it was measured under was `--same-text --max-steps 64`:

| | the C80 soak | a realistic bank |
|---|---|---|
| texts | one, repeated | `tests/load_texts_en_v2.txt`, 273 sentences |
| cap | `--max-steps 64` | none |
| audio per request | **1.65 s**, sd 0.56 | **4.48 s**, sd 4.48 |
| longest | ~5 s | 19.4 s |

A standard deviation equal to the mean is the point: a ragged mix of 1 s and
19 s utterances is a different scheduling problem from a uniform one, because a
long stream must be fed without a gap for twenty seconds while 79 others are in
flight. Aggregate throughput barely moved (88.4 vs 58.4 audio-s per wall-s);
what broke was per-stream continuity, which is the half a listener hears.

I chose `--same-text --max-steps 64` so the topology arms would be comparable to
each other. They were. The error was promoting that number to a serving capacity
without removing the simplification that existed only for the comparison -- and
the harness had warned, at every level, that the class mix was not identical.

## What IS established (measured, quiet box, 10 minutes)

`prefork 12 x 2`, C80, **uniform 1.65 s utterances**, 33,899 launched and
33,899 completed:

| gate | value | threshold |
|---|---|---|
| STREAM_RTF p95 | 0.880 | < 1.00 mandatory, <= 0.90 preferred |
| stall @500 ms / @250 ms | **0 of 33,899** | == 0 |
| TTFB p95 | 70.5 ms | <= 100 |
| TTFA p95 | 183.8 ms | <= 500 |
| required prebuffer p95 | 13.3 ms | <= 500 |

Ten 60 s windows, last-vs-best drift **+0.0030** RTF and **+10 ms** prebuffer
against tolerances of +0.050 and +150 ms. It does not degrade.

Topology screen at C<=24 (all seven cuts, 3 waves each), STREAM_RTF p50:
1x24 **0.568** (safe_play_start p95 2533 ms), 3x8 0.490, 4x6 0.363, 6x4 0.298,
12x2 0.273, 24x1 0.237. Monotone in WORKERS, not threads. `doctor.py` predicted
6x4 from "the AR step stops scaling past ~4 threads"; that is true of a single
request and is the wrong lever for serving.

**Admission is 8 slots per worker and it is arithmetic, not silicon.** Every
refusal count matched `(C - (8W + W)) * waves` exactly: 2x12 refused 18 of 72 at
C24; 6x4 refused 126 of 288 at C96; 12x2 refused 108 of 432 at C144; 24x1
refused none at C144. Sizing on cores alone mispredicts this.

## What is NOT established

- **Capacity under a realistic corpus.** The only evidence is CONFOUNDED and
  must not be quoted: the bank-driven generator that ran during the audio
  capture reported C79 failing every mandatory gate (STREAM_RTF p95 1.478,
  4659 of 8216 stalled at 500 ms) -- but it ran WITH the capture process
  alongside it, on a box at loadavg 32. Two variables moved. The clean
  experiment is a bank soak with nothing else running, and it did not happen:
  I launched it, reported it as running, and never checked that it had produced
  a single line before the box went away. It had not.
- **Cadence as a certified number.** The load generator shares the 24 cores;
  19% of its socket reads returned already-queued data, above the harness's own
  15% refusal threshold. Every cadence percentile here is an upper bound on
  server lateness. Axion measured ~13% on 32 cores and stayed under it. The fix
  is a second host for the generator, not a different split of this one --
  confining the client to 4 cpus made it WORSE (46.1%).
- **The zero-stall result is the exception and survives.** Coalesced reads make
  measured gaps larger, never smaller, so zero stalls under coalescing stays
  zero without it.
- **Audio quality.** Not scored, at all.
- **Thirty-minute behaviour.** Longest run was ten minutes.

## The client deliverables

- `~/Desktop/PocketTTS-x86-Genoa-Client-Report-2026-09-22.INVALID-do-not-send.pdf`
  -- generated, then invalidated by the finding above. Regenerate with
  `tools/client_report_genoa_x86.py` once a bank-driven number exists.
- `/root/bundle80.zip` on the box, 48 WAVs, 10.4 MB, four length classes,
  223 s of audio, captured at a CONFIRMED loadavg of 32 on 24 cores. **Never
  downloaded before the box was released.** The audio was valid; the capacity
  figure printed beside it was not.

## Defects this box found, all fixed and pushed

| commit | defect | invisible on |
|---|---|---|
| cba879a | `isa.arm.bf16` read `compiled no` while VDPBF16PS executed; declared no env while `MYNAH_QMAT_BF16` moved it; printed `IDLE HARDWARE` on an AMD CPU | any host without AVX512-BF16 |
| cba879a | `doctor.py` dropped the reason, so an operator saw `isa.arm.bf16 ON` on an EPYC with no explanation | aarch64 |
| cba879a | `x86-tier-parity` allowlist predated "resolved means runs"; FAILED on any AVX-512 host with a pack | CI (no model pack) |
| 30db397 | CodeQL #5, CWE-134: `MYNAH_POOL_METER_JSON` used as a format string. Verified to SIGSEGV | nothing calls it; it is a public symbol only |
| edcd64e | prefork refusals counted only to stderr: `/health` said `rejected: 0` while the router answered 18 of 72 with 503 | a single-process server |
| edcd64e | `capture_bundle.sh` hardcoded the 32-core Axion topology while calling itself generic | Axion |
| (local) | `capture_bundle.sh` did not check the load generator survived: it refused to start, and 48 files were captured on an IDLE box while the script printed `loadavg 1.73` beside the words "load at C80" | a box where the generator happens to start |

Also: gcc 13.3 emits 23 warnings clang does not, 7 of them
`-Waggressive-loop-optimizations` in `src/kernels.c` and `src/qmat.c`
("iteration 4611686018427387903", i.e. `SIZE_MAX/sizeof(float)`). Unfixed; no
call site can produce such an `n`, but they sit in the hottest files on the
production compiler and cover real ones.

## Next session, in order

1. **Bank soak, nothing else running.** `--bank tests/load_texts_en_v2.txt`, no
   `--max-steps`, screen 16/24/32/48/64 then soak 10 min on the survivor. Rough
   expectation from the confounded run, to be confirmed or killed: somewhere
   between C32 and C48.
2. Only then regenerate the client report, and download the bundle FIRST.
3. Write `configs/perf/epyc-9254-24c-pocket-en.json` so the next x86 soak is one
   command, the way Axion's is.
4. The 7 gcc warnings; the candidate is a zero-cost
   `if (n > PTRDIFF_MAX / sizeof(float)) __builtin_unreachable();`.
