# `.work/` — task detail notes

`PLAN.md` is the board: one line per work item, with a link to the note here
that holds the detail. Nothing in `PLAN.md` should need more than one line.

## Rules

1. **One note per work item.** Name it after the item, not after a date:
   `engine-seam-refactor.md`, not `2026-09-12-refactor.md`.
2. **A note is written before the work starts**, not after. It states the
   problem, the evidence, the plan, and the acceptance gate. If it cannot state
   a gate, the item is not ready to be worked on.
3. **A note is falsifiable.** Record what was measured, on what machine, with
   what flags. Record rejected ideas and *why* they were rejected — a negative
   result that is not written down gets re-tried.
4. **`PLAN.md` never grows a log.** Session checkpoints, measurements and
   post-mortems live here. Historical measurements that are still true belong in
   `docs/performance.md`; this folder is for work in flight and for the reasoning
   behind decisions.
5. **Close a note, do not delete it.** Mark the status at the top
   (`OPEN` / `IN PROGRESS` / `DONE <date>` / `REJECTED <date>`) and leave it.

## Index

| Note | What it covers |
|---|---|
| [pocket-tts-model-facts.md](pocket-tts-model-facts.md) | Verified facts read out of the PocketTTS weights |
| [pocket-tts-vs-mynah.md](pocket-tts-vs-mynah.md) | Architecture diff: mynah-tts as-is vs PocketTTS |
| [engine-seam-refactor.md](engine-seam-refactor.md) | E1 — the second-engine seam, splitting `graph.c` |
| [pocket-tts-oracle.md](pocket-tts-oracle.md) | E2 — per-stage Python oracle and tolerances |
| [pocket-tts-engine.md](pocket-tts-engine.md) | E3 — the PocketTTS engine in C |
| [cpu-kernels-arm-x86.md](cpu-kernels-arm-x86.md) | E4 — CPU kernel layer, ARM and x86 in one step |
| [streaming-server-v2.md](streaming-server-v2.md) | E5 — streaming server v2, ported from qwen-tts |
| [licensing-and-voice-policy.md](licensing-and-voice-policy.md) | E6 — per-voice licensing in the model pack |
| [archive-2026-07-checkpoints.md](archive-2026-07-checkpoints.md) | Historical `PLAN.md` §17-24, moved verbatim |
