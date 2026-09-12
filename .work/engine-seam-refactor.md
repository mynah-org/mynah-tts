# E1 — Engine seam and the `graph.c` split

Status: **OPEN** · blocks E3 · does not block E2

Context: [pocket-tts-vs-mynah.md](pocket-tts-vs-mynah.md) §4.
This is the July TODO at `PLAN.md` "Architecture TODO — streaming/server split",
now with a concrete second model to shape it.

## Problem

There is no seam. `src/tts_engine.h` and `src/engine_magpie.c` are designed in
the plan and named in `CLAUDE.md`, but do not exist. Everything lives in
`src/graph.c` (4283 LOC), the public header carries Magpie's dimensions, and the
JSON parser cannot represent a per-engine config. A second engine cannot be
added beside the first; it can only be added *after* the seam exists.

Do the refactor **with PocketTTS in hand, not before**. The only way to find out
which abstraction is real is to have two engines that must fit it. Guessing the
seam from one engine is how the current situation happened.

## Target shape

```
src/tts_engine.h        vtable: prepare / step / flush / reset / free + caps
src/engine_magpie.c     Magpie graph, moved out of graph.c unchanged in behaviour
src/engine_pocket.c     E3
src/inference.c         slot driver, batching, RNG, EOS  (from graph.c:3674-4283)
src/stream.c            streaming state, chunk emission  (from graph.c:3656-3968)
src/transformer.c       attention, FFN, KV, RoPE, norms  (from graph.c:1-713)
src/json.c              a real parser: nesting, arrays, scopes
```

`graph.c` should not survive the split as a file.

## Work items

1. **`tts_engine.h`** — the vtable, plus a capability block the engine fills in
   (`is_discrete_codec`, `frames_per_step`, `sample_rate`, `latent_dim_or_0`,
   `needs_cfg`). The driver must never ask "which engine is this?".
2. **Move the driver out.** `graph.c:3674-4283` is already close to
   model-agnostic: it assumes only "AR steps produce something an audio decoder
   consumes". Lift it to `inference.c` and make the step call go through the
   vtable. **Behaviour must not change** — the Magpie stream/offline parity test
   is the gate.
3. **Move streaming out.** `stream.c` owns `STREAM_CONTEXT_FRAMES`-style bounded
   suffix decoding, but the constant becomes an engine capability: Magpie's 32
   NanoCodec frames and PocketTTS's SEANet receptive field are different numbers.
4. **Clean the public header.** Remove `codebook_count`, `codebook_size`,
   `frame_stacking_factor`, `audio_bos_id`, `audio_eos_id`,
   `local_transformer_layers`, `use_local_transformer` from `src/mynah_tts.h`.
   They go behind an opaque per-engine blob. This is a **breaking API change**;
   `cli/main.c` and `server/main.c` only print them today, so the blast radius is
   small — check before assuming.
5. **Replace the JSON parser.** `src/mynah_tts.c:112-163` is `strstr` scanning.
   Needs nesting and arrays for `"engine": {...}` and for `[6,5,4]`.
6. **Generalize the converter.** `tools/convert_magpie.py:139-144` writes
   Magpie's constants as literals. Split into a shared pack writer plus a
   per-engine metadata producer.

## Acceptance gate

- `make` and `make self-test` clean on macOS arm64 and Linux x86-64.
- Magpie **stream↔offline parity is sample-identical** to the pre-refactor
  binary — capture goldens *before* starting.
- `tests/test_server.sh` fully green, including the batch/stream parity and
  concurrency checks.
- `make ubsan` clean; `make leaks` clean on macOS, since ownership moves.
- `grep -c 'codebook\|frame_stacking\|local_transformer' src/mynah_tts.h` → 0.
- No file in `src/` over ~1200 LOC.

## Risks

| Risk | Mitigation |
|---|---|
| Refactor silently changes Magpie audio | byte-compare goldens captured before the split; this is the one non-negotiable gate |
| The seam is shaped around Magpie again | do not merge E1 before `engine_pocket.c` compiles against it, even as a stub |
| Ownership bugs when state moves modules | `make leaks` + `make ubsan` in the same change, not after |
| Scope creep into kernel work | E4 is a separate epic; touch no kernel here |
