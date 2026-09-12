# E2 — PocketTTS per-stage oracle

Status: **IN PROGRESS** — `tools/oracle_pocket.py` written and running (2026-09-12); `tests/parity_pocket.py` still to do

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

## First run — 2026-09-12

`make oracle-pocket` (english / alba / seed 1234), 132 tensors, 26 tokens,
52 frames, 4.16 s of audio, `temp=0.3`, `sampler_decode_steps=1`,
`eos_threshold=-4.0`. Dumps land in `build/oracle-pocket` (gitignored).

### The captured chain, one AR step

```
conditioner.embed   in [1,0]         out [1,0,1024]   # text only at prefill
input_linear        in [1,1,32]      out [1,1,1024]   # previous latent
transformer.layers  in [1,1,1024]    out [1,1,1024]   # 6 layers
out_norm            in [1,1,1024]    out [1,1,1024]
out_eos             in [1,1024]      out [1,1]        # scalar logit vs -4.0
flow_net            in0 cond [1,1024]                 # = out_norm result
                    in1 s    [1,1]  = 0
                    in2 t    [1,1]  = 1
                    in3 x_0  [1,32]                   # the gaussian noise
                    out      [1,32]                   # the latent
mimi.quantizer      in [1,32,1]      out [1,512,1]    # after denorm + transpose
mimi.upsample       in [1,512,1]     out [1,512,16]   # 12.5 Hz -> 200 Hz
decoder_transformer in [1,512,16]    out [1,512,16]   # inner layers see [1,16,512]
mimi.decoder        in [1,512,16]    out [1,1,1920]   # exactly one 80 ms frame
```

### What this confirmed, beyond the shapes

- **`s = 0`, `t = 1`, one call.** The LSD head really is a single evaluation per
  frame, with the two time conditions pinned at the endpoints. There is no
  integration loop to write.
- **The noise is captured** (`flow_net.in3`), which is what makes stage 7
  comparable at all. Without it the flow head can only be checked
  distributionally.
- **The decoder transformer runs at the encoder frame rate**, on 16 positions per
  12.5 Hz frame, and its inner layers see the tensor transposed to `[1,16,512]`.
  The C implementation must place the transpose in the same place.
- Between `flow_net.out` and `mimi.quantizer.in` sit the latent denormalization
  (`emb_mean`/`emb_std`) and a transpose. Stage 8 is therefore checkable as the
  delta between two captured tensors rather than needing its own hook.

### Bug found and fixed in the tool itself

The first version dropped **every input tensor** and every module whose output is
a tuple or list (`ProjectedTransformer` returns one), because `to_numpy`
recursed over containers and then re-entered with an already-converted
`ndarray`, which it did not recognise and returned `None` for. 48 tensors
captured instead of 132. Fixed by an ndarray passthrough, plus `attach()` now
**fails loudly** when a watched module name no longer exists upstream rather than
silently recording less.

Worth stating because it is the failure mode this whole epic exists to prevent:
a dump that looks fine and is quietly incomplete.

### Still to do

Stages 3 (voice KV as loaded) and 11 (per-stage SEANet conv outputs) are not
individually hooked yet, and none of the streaming-specific measurements
(receptive field, `context: 250` eviction, chunk seam) have been run. Those are
E2-3 to E2-5 and they are the ones that feed E1's design.
