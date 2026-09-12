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

## E2-3 measured — 2026-09-12: the codec must carry state, not replay context

`tools/oracle_pocket_stream.py`, english/alba, 52 frames, chunk sizes 1-16.
Two candidate designs for `stream.c`, compared against a one-shot decode:

**A. carry codec state across chunks** (what upstream does)

| chunk | max abs error | rel RMS |
|---|---|---|
| 1 frame (80 ms) | 1.0e-06 | 7.3e-07 |
| 2 | 1.2e-07 | 3.4e-08 |
| 4 | 1.2e-07 | 2.4e-08 |
| 16 | 6.0e-08 | 1.1e-08 |

Float32 rounding, nothing more. **Exact for practical purposes down to a single
80 ms frame.**

**B. fresh state per chunk, replaying K frames of left context** (what the
Magpie/NanoCodec path does today with `STREAM_CONTEXT_FRAMES 32`)

| K frames | max abs | rel RMS | |
|---|---|---|---|
| 0 | 1.13 | 7.2e-01 | unusable |
| 8 | 3.1e-01 | 1.2e-01 | |
| 16 | 1.2e-01 | 3.8e-02 | |
| 32 | 4.0e-04 | 6.4e-05 | audible-threshold, not exact |
| **64** | **0.0** | **0.0** | **bit-identical** |

**Decision: option A.** Option B needs **64 frames = 5.12 s** of replayed
context per chunk. At Magpie's 21.5 fps its 32 frames are ~1.5 s; the PocketTTS
equivalent is over three times longer in wall-clock and, because each replayed
frame costs a full SEANet pass, roughly an order of magnitude more work per
emitted chunk. It is not a viable streaming design here.

### The trap that made this measurable

The first run said option A was *wrong* — rel RMS 0.46, worse as the chunk got
smaller. It was not: `increment_steps(mimi, state, stride * n_latents)` after
every `decode_from_latent` was missing from the harness
(`models/tts_model.py:537`). The codec keeps an **explicit position counter,
separate from the convolution ring buffers**, advancing by `encoder_stride` (16)
per latent frame. Without it every chunk rewrites the decoder transformer's KV
from position zero.

It fails *gracefully*: no crash, no warning, output that degrades smoothly with
chunk size. A C implementation will hit exactly this, and the symptom will look
like a kernel bug rather than a missing counter. Two pieces of state, both
mandatory:

1. convolution ring buffers (the obvious one)
2. a position counter advanced by `encoder_stride` per latent frame (the one
   that gets forgotten)

Also confirmed here: the codec's `sequence_length` is sized in **encoder frames**
(`frames * 16`), not latent frames. Sizing it in latent frames fails loudly, at
least — the first chunk's KV write raises.

### Consequence for the engine seam

`caps.audio_left_context_frames` (proposed in
[engine-seam-refactor.md](engine-seam-refactor.md)) is the wrong abstraction: it
assumes replay. Make `decode_audio` require **contiguous, monotonically
increasing** frame ranges and let the engine keep whatever state that needs —
Magpie replays 32 frames internally, PocketTTS carries a ring buffer and a
counter. The capability then disappears from the public surface instead of
leaking one engine's strategy into the driver.
