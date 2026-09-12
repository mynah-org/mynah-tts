# E2 — PocketTTS per-stage oracle

Status: **OPEN** · independent of E1 · **do this first**

`CLAUDE.md`: *"Every new stage gets a Python oracle dump before downstream work
depends on it."* This epic is that dump, and it is cheap: the reference
implementation is pip-installable and CPU-only.

Doing it before E1 is deliberate — the dumps tell us which pieces of state the
engine seam actually has to model, instead of guessing.

## Setup

```bash
uv run --with pocket-tts python tools/oracle_pocket.py ...
```

No GPU, no build. Reference source is cloned at `/tmp/pocket-tts-src`
(`kyutai-labs/pocket-tts`, MIT) — re-clone rather than rely on that path.

## Stages to dump

Fix seed, temperature, voice, text and decode steps for every run and record
them in the dump header.

| # | Stage | Tensor to dump | Tolerance |
|---|---|---|---|
| 1 | tokenizer | token IDs for a UTF-8 + punctuation + digits corpus | **exact** |
| 2 | LUT conditioner | `text_embeddings` `[1, n_tok, 1024]` | abs 1e-5 |
| 3 | voice prefix | KV cache as loaded, per layer | **exact** (it is just a file) |
| 4 | backbone prefill | post-`out_norm` hidden `[1, T, 1024]` | abs 1e-4 / rel 1e-3 |
| 5 | backbone step | hidden for one AR step, steps 0/1/2/N | abs 1e-4 / rel 1e-3 |
| 6 | EOS | `out_eos` logit per step, and the step where it crosses `-4.0` | **exact step index** |
| 7 | flow head | `x_0` noise in, latent out, 1 LSD step, fixed noise | abs 1e-4 |
| 8 | latent denorm | after `emb_mean`/`emb_std` | abs 1e-5 |
| 9 | Mimi input | `quantizer.output_proj` + `upsample` output | abs 1e-4 |
| 10 | decoder transformer | 2-layer output `[1, T', 512]` | abs 1e-4 |
| 11 | SEANet | per-stage conv output, all 3 convtr stages | abs 1e-3 |
| 12 | waveform | full WAV | duration exact; peak/RMS 1%; mel corr > 0.99 |

Stage 7 needs the noise injected from outside — patch the RNG or pass the noise
tensor in, otherwise nothing downstream is comparable.

## Streaming-specific dumps

These decide the design of `stream.c`, so they are not optional:

- **SEANet receptive field**: the minimum left context in frames for which
  chunked decoding is sample-identical to one-shot decoding. This is the
  PocketTTS equivalent of `STREAM_CONTEXT_FRAMES 32` and must be measured, not
  assumed.
- decoder-transformer behaviour at `context: 250` once generation exceeds 250
  frames (20 s) — what is evicted, and does output change.
- `MAX_TOKEN_PER_CHUNK = 50` chunking: dump the seam between two text chunks and
  check for discontinuity. Upstream has a known text-skipping bug here; record
  what the reference actually does so we reproduce it rather than "fix" it.

## Deliverables

- `tools/oracle_pocket.py` — dumps every stage above to `.npy` + a JSON manifest
  (model revision, sha256, seed, params).
- `tests/parity_pocket.py` — compares a C dump against the oracle with per-stage
  tolerances, exit non-zero on failure.
- `make oracle-pocket` and a `make test-parity-pocket` that a human can run.
- A short results table appended to this note once it runs.

## Acceptance gate

All 12 stages dumped and reproducible across two runs with the same seed. The
SEANet receptive field is a **number written in this note**, not an assumption.
