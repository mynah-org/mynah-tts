# Serving profiles

**One question: given this hardware and this engine, how should the server be run — and
what has to be true before that answer may be called qualified?**

```
configs/perf/schema.json                  the format
configs/perf/axion-c4a-32c-pocket-en.json GCP Axion c4a, 32 cores, PocketTTS English
configs/perf/recommended.json             stable entry point; POINTS at the current one
```

```bash
tools/perf_profile.py validate                                  # every profile
tools/perf_profile.py best     recommended                      # what it has qualified
tools/perf_profile.py command  recommended --model models/pocket-en --port 8080
tools/perf_profile.py soak     recommended --model models/pocket-en
tools/perf_profile.py new      my-box --like axion-c4a-32c-pocket-en
python3 tests/test_perf_profile.py
```

And the one that matters, because it makes the configuration unforgeable by hand:

```bash
python3 tools/serving_profile.py --profile axion-c4a-32c-pocket-en \
        --model models/pocket-en --server-bin build/cpu/mynah-tts-server
```

That single flag supplies the worker topology, the environment, the bank, the soak
length and every gate threshold — and **refuses to run** if the environment or the
command line contradicts any of them.

## Why this exists, precisely

Between 2026-09-13 and 2026-09-19 every capacity number in `docs/performance.md` was
measured with `MYNAH_QUANT_GROUPS=...,codec_convtr:int8` exported by a shell line, while
the shipped binary chose something 1.86x slower on the per-slot term. Both facts were
written down. Neither was wrong. But the harness could not see the difference, so the
runs passed, the documentation described an operating point, and **the product could not
reach it by itself**. Nothing caught that for six days.

A profile turns that class of gap into a validation failure:

* a variable the profile declares `null` must be **absent** from the environment, not
  merely unset by the harness — a leftover `export` refuses the run by name;
* `runtime.matches_shipped_default: true` asserts that the profile exports *nothing*,
  which is the only way a soak measures what a plain binary does;
* a flag also given on the command line is a **conflict, not an override**: a profile
  whose values can be quietly replaced qualifies nothing.

## What is a profile, and what is not

| | contains | example |
|---|---|---|
| **profile** | how the server should be run, and the gates | `16 workers x 2 threads, max-batch 8, TTFA p95 <= 500 ms` |
| **run artifact** | what exactly was measured that once | binary sha, seed, timestamp, the full report under `reports/` |

A profile carries the operating point it earned and a pointer to the evidence, but not
the provenance of one run — that is what lets it survive a new binary.

## The rules the validator enforces, and why each was paid for

| rule | the mistake it prevents |
|---|---|
| `status: qualified` needs a soak of at least `gates.soak.seconds` | a ten-minute screen promoted C90 on a 4.6 ms margin; the thirty-minute run moved the `required_prebuffer` **maximum** 495 → 535 ms while its p95 did not move at all. A percentile is stable in sample size; a maximum is not |
| `qualified` requires verdict GOOD and zero stalls | "almost GOOD" is MARGINAL, and MARGINAL is not a product point |
| `history` must END at `operating_point` | so "the best" lives in one place and a regression is visible as a step backwards |
| a GOOD level may not sit in `ceiling` | C96 was GOOD for a day before anyone looked above C90: an unexamined ceiling is capacity thrown away |
| `preferred_concurrency` must equal the measured level | the number a deployment reads must be the number a soak earned |
| `prefork_workers x threads_per_worker <= logical_cpus` | oversubscription measured as a scheduling failure, not as a model result |
| `server.device: cuda` needs `hardware.gpu`, `http_workers`, `max_inflight >= max_batch`, and no prefork | the CUDA server is one process with one scheduler thread; a `max_inflight` below `max_batch` is silently lowered by the server, so the profile would not say what runs |

## Starting a new box

`tools/perf_profile.py new <id> --like <reference>` writes a skeleton with the hardware
blanked and `status: unqualified`. Fill it from what **that** machine reports, then run
the soak and write the point it earned.

Do not inherit the topology. `16x2` is right for PocketTTS (109.5M) and `4x8` is right
for qwen-tts (1.7B) **on the same 32-core host**: the optimum follows the model's
`T_frame(B) = a + b*B` ratio, not the machine. Re-screen it.

## GPU boxes

`server.device` is `cpu` when absent, so every profile above is unchanged. A `cuda`
profile is one process — `http_workers` HTTP threads in front of one scheduler thread
that owns the GPU — and carries `hardware.gpu`; `hardware.architecture`/`logical_cpus`
then describe the host. `recommended.json` stays the CPU recommendation.

```
configs/perf/l4-24g-pocket-en-24l-cuda.json  one NVIDIA L4 (sm_89), Pocket 24L: C160 qualified, 184.5 audio-s/s
configs/perf/l4-24g-pocket-en-6l-cuda.json   one NVIDIA L4 (sm_89), Pocket 6L:  C256 qualified, 316 audio-s/s
```

```bash
tools/l4/provision.sh sm_89       # on the box: builds `make cuda-server cuda`, converts both packs
tools/perf_profile.py command l4-24g-pocket-en-24l-cuda --model models/pocket-english-24l --port 8080
tools/perf_profile.py command l4-24g-pocket-en-6l-cuda  --model models/pocket-english-6l  --port 8080
tools/perf_profile.py soak    l4-24g-pocket-en-24l-cuda --model models/pocket-english-24l
```

`command` prints the exported variables and `build/cuda/mynah-tts-server --device cuda -w
... --max-batch ... --max-inflight ...`. `soak` does not start the CPU harness for a cuda
profile (and `tools/serving_profile.py --profile` refuses one): it prints the
`tools/l4/qualify.sh` run — two 30-minute soaks on fresh servers, audio captured from
the live streams, WER, evidence bundle — under `tools/l4/detach.sh`, plus the single
`tools/l4/soak.sh` screen. The harness starts the server through `tools/l4/serve.sh`,
and `tests/test_perf_profile.py` checks that its exports are exactly the profile's.

Neither L4 profile is the shipped default: `MYNAH_CUDA_KV_DTYPE=bf16` and
`MYNAH_THREADS=1` are required. Every `MYNAH_CUDA_*` switch that is on by default is
declared `null` with its measured win and what `=0` rolls back to, and so are the
opt-ins that must stay off (`MYNAH_CUDA_PREFILL_FIXED=0` is +4-9% but loses
piecewise-prefill identity).
