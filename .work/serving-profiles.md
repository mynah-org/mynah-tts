# E10-17 — a serving configuration that cannot be got wrong by hand

Item: `PLAN.md` E10-17. Files: `configs/perf/{schema.json,README.md,*.json}`,
`tools/perf_profile.py`, `tools/serving_profile.py` (`--profile`),
`tests/test_perf_profile.py`.

## Problem

Between 2026-09-13 and 2026-09-19 every capacity number in `docs/performance.md`
was measured with `MYNAH_QUANT_GROUPS=...,codec_convtr:int8` exported by a shell
line, while the shipped binary chose something 1.86x slower on the per-slot
term. Both facts were written down, in the docs and in `PLAN.md` E10-14. Neither
was wrong. The harness could not see the difference, so every run passed, the
documentation described an operating point, and **the product could not reach it
by itself**. It took six days and a direct question to notice.

That is not a documentation failure, it is a missing gate. The configuration of
a qualifying run lived in shell history, where nothing could check it.

A second, quieter cost: every new box started from a paragraph in a `.work`
note, so "what do I run on a 32-core ARM host" was answered by reading prose and
retyping flags — the step where a value silently becomes the wrong value.

## What was built

A **profile**: a JSON file that carries the deployment configuration *and* the
gates *and* the operating point that was measured on that hardware.

    configs/perf/schema.json                    the format
    configs/perf/axion-c4a-32c-pocket-en.json   GCP Axion c4a-standard-32, PocketTTS EN
    configs/perf/recommended.json               an alias; points, never copies

`tools/serving_profile.py --profile <id>` then supplies the topology, the
environment, the bank, the soak length and every threshold — and refuses when
the world disagrees:

* a variable the profile declares `null` must be **ABSENT** from `os.environ`, not
  merely unset by the harness. A leftover `export MYNAH_QUANT_GROUPS=...` names
  itself in the refusal. This is the exact six-day bug, turned into an error;
* `runtime.matches_shipped_default: true` asserts the profile exports *nothing*,
  which is the only configuration in which a soak measures what a plain binary
  does;
* a flag also given on the command line is a **conflict, not an override**. A
  profile whose values can be replaced from the shell qualifies nothing.

## The validator encodes the mistakes we actually made

| rule | the mistake |
|---|---|
| `qualified` needs a soak of at least `gates.soak.seconds` | a ten-minute screen promoted C90 on a 4.6 ms margin; the thirty-minute run moved `required_prebuffer` MAX 495 → 535 ms while its p95 did not move. A percentile is stable in sample size, a maximum is not |
| `qualified` requires GOOD and zero stalls | MARGINAL is not a product point |
| `history` must END at `operating_point` | "the best" in one place; a regression reads as a step backwards |
| no GOOD level in `ceiling` | C96 was GOOD for a day before anyone measured above C90. An unexamined ceiling is capacity thrown away |
| `preferred_concurrency` == the measured level | the number a deployment reads is the number a soak earned |
| `workers x threads <= logical_cpus` | oversubscription measured as if it were a model result |

## Answering "what is this box" in one command

`tools/perf_profile.py detect` reads the host (architecture, cores, SMT, NUMA,
RAM, ISA flags) and ranks the profiles for that architecture, with architecture
as a **wall rather than a weight**: an ARM profile on an x86 box is not "a bit
off", it is a different set of kernels.

An exact hardware match prints "reproduce it before believing it" — same silicon
is not the same machine. Anything else prints the four steps to a new profile
and one refusal: **do not inherit the topology.** `16x2` is right for PocketTTS
(109.5M) and `4x8` is right for qwen-tts (1.7B) on the *same* 32-core host,
because the optimum follows the model's `T_frame(B) = a + b*B` ratio, not the
machine's core count.

## Acceptance gate

`python3 tests/test_perf_profile.py` — the committed profiles validate, the alias
resolves, the resolution produces the right argv and thresholds, and each
semantic rule is broken on purpose and caught. The half that matters is the last
block: a forbidden variable in the environment and a conflicting command-line
flag must each refuse a run.

## Prior art, and what was deliberately not copied

The shape comes from `qwen-tts` `configs/perf/`, which is the same idea carried
much further: parity lanes, `[FLAGS]` declaration checks, per-ISA product
profiles, a `doctor` that predicts a topology from measured roofs. Two things
were left out on purpose.

**The forty-eight environment variables.** That file has a large `runtime.environment`
because that engine has that many knobs. Ours has three
(`MYNAH_QUANT_GROUPS`, `MYNAH_PREFILL_SLICE`, `MYNAH_PREFILL_STEP_MS`) and after
2026-09-19 all three are declared *absent* in the Axion profile, because the
shipped default now covers them. **A profile that exports nothing is the
strongest one that can be written** — it cannot drift from the binary, because it
is the binary. Every variable added here is a claim the default is wrong.

**Parity lanes.** They answer "is the intended kernel actually active", which
matters when one engine has AMX, VNNI, KAI and Design-D paths that all produce a
number. This engine has one CPU path per ISA and a self-test that covers it.
When the x86 kernel stops being unexecuted, revisit.
