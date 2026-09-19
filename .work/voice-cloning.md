# E7 — Voice cloning

Status: **IMPLEMENTED** 2026-09-12 — `src/voice_clone.{c,h}`, 2555 LOC, in `CORE_SOURCES`
and `--self-test`. Parity against the PyTorch oracle to 6.4e-06.

Earlier notes treated cloning as optional and proposed shipping a pack without
the encoder. **That is reversed**: cloning is a requirement, the encoder ships,
and the pack must be built from the official gated Kyutai repo.

## Why this is cheap on a latent model

Cloning is natural here in a way it is not on a discrete-codec model. The voice
condition and the generated output live in **the same 32-dim latent space**: the
encoder turns a reference wav into exactly the kind of latents the model produces
itself, and the backbone is prefilled with them. There is no separate speaker
encoder, no x-vector, no CAM++, no adapter. The "voice" is just the KV cache left
behind by having read some audio.

Cost: `mimi.encoder*` + `mimi.downsample` + `speaker_proj_weight` =
**9.78M params, 19.6 MB BF16**, already present in every current-generation
checkpoint (verified on `english_2026-04`).

## The path, as upstream implements it

```
wav  -> decode, to mono, resample to 24 kHz, truncate to 30 s
     -> mimi.encoder (SEANet, mirror of the decoder)
     -> encoder_transformer (2 layers, d512, layer_scale)
     -> ConvDownsample1d(stride 16, kernel 32, pad "replicate")  -> latents [1, T, 32]
     -> F.linear(latents, speaker_proj_weight)                   -> [1, T, 1024]
     -> cat([bos_before_voice, prompt])       if insert_bos_before_voice
     -> prefill through flow_lm               -> KV cache = the voice
     -> (optional) serialize the KV as a .safetensors voice file
```

`_encode_audio` in `models/tts_model.py:443`; `get_state_for_audio_prompt` at
:885.

## What is genuinely new C code

| Item | Note |
|---|---|
| SEANet **encoder** | mirror of the decoder: plain conv1d where the decoder has transposed conv. Same kernels, different direction. Cheap once E3-4 exists. |
| encoder transformer | 2 layers, d512, `layer_scale` — identical shape to the decoder transformer already needed in E3. Reuses shared attention. |
| `ConvDownsample1d` | stride 16, kernel 32, **`pad_mode="replicate"`** — note it is *not* zero padding. |
| audio input | WAV decode. Start by **requiring 24 kHz mono**; that defers the resampler without blocking the feature. |
| **resampler** | the one real piece of new DSP: `scipy.signal.resample_poly(up, down)`, i.e. rational polyphase FIR. Match it within tolerance against the oracle rather than trying to be bit-exact. |
| `export-voice` | serialize the KV. Trivial, and it makes reload instant. |

Note `ConvTrUpsample1d` (the decode-side counterpart, needed anyway in E3) uses
**`groups=dimension`** — it is a *depthwise* transposed conv, which is why the
weight is `[512, 1, 32]` and not `[512, 512, 32]`. Do not implement it as a dense
transposed conv.

## Version differences live only here

`speaker_proj_weight` is `[1024, 512]` in `english_2026-01` and `[1024, 32]` in
the current generation, and `bos_before_voice` exists only in the current one.
Both are config-driven: the projection's input dim comes from `model.json`, and
`insert_bos_before_voice` is a boolean. No second code path.

## Consent — design it in, not on

The prohibited-use clause with actual force is *"voice impersonation or cloning
without explicit and lawful consent"*. With cloning shipped this stops being a
licensing footnote. Required before the feature is exposed:

- an explicit affirmation of consent before cloning from a supplied file, in both
  the CLI and the API, recorded rather than assumed
- a notice in `--help`, in the README and in the API error text
- **no default that clones silently**

See [licensing-and-voice-policy.md](licensing-and-voice-policy.md). Separately,
EU AI Act art. 50 transparency duties apply to a product that ships this.

## Acceptance gate

- A cloned voice from a 24 kHz mono wav produces a KV cache that matches the
  oracle's within the E2 tolerances.
- The same wav resampled from 44.1 kHz and 48 kHz lands within tolerance of the
  24 kHz path.
- `export-voice` round-trips: exported KV reproduces the audio of the in-memory
  one, sample-identical.
- A voice file is **rejected** when it does not match the loaded model revision
  (upstream: such a model "typically never emits EOS").
- Cloning without an explicit consent affirmation is refused.
- `make leaks` and `make ubsan` clean on the cloning path.

## Implemented — 2026-09-12

`src/voice_clone.{c,h}`. It is **composition, not reimplementation**: every
encoder convolution goes through `mynah_causal_conv1d` from `seanet.h` (which
already supported stride, dilation, groups and replicate padding), the
downsample through `mynah_seanet_downsample_*`, and the encoder transformer and
backbone prefill through `mynah_transformer_ar_*`. `seanet.c` was not touched.

### Parity against the PyTorch oracle, not just structure

| stage | max relative error |
|---|---|
| Mimi latents `[52,32]` | 2.4e-06 |
| KV layers 0-5 | 1.2e-06 … 6.4e-06 |
| same path from a 48 kHz wav (resampler included) | 2.1e-06 … 6.2e-06 |
| resampler vs `scipy.signal.resample_poly` 48k→24k | **2.4e-07** |
| chunked (1, 4, 32 frames) vs unchunked KV | **byte-identical** |

The encoder runs in chunks by default because 30 s at 24 kHz over 64 channels
would be 184 MB of activations; the conv ring buffers and transformer KV make
chunking exact rather than approximate.

### The `fantine` discrepancy, resolved by experiment

Cloning from `vctk/p244_023.wav` produces 132 frames + BOS = 133, **exactly the
shape of the distributed `fantine.safetensors`** — so it is the right wav. The
*values* differ: mean abs 0.023-0.070 per layer, correlation 0.9927-0.9994.

The decisive experiment was to run the **upstream PyTorch oracle on the same
wav**. It diverges from the distributed file by *exactly the same amount* (layer
0: max 1.024, mean 2.317e-02, corr 0.999028 — digit for digit the same as our
C), while our C matches PyTorch to **7.9e-06**.

So the difference is in whatever preprocessing Kyutai used when generating the
shipped embedding — loudness normalization, a different crop, or a different
revision of the wav — **not a bug here**. Worth keeping: "our output differs
from the reference artifact" and "our output differs from the reference
implementation" are different claims, and only the second is a defect.

### Consent is in the API, not in a document

`mynah_voice_clone_consent` is required and explicitly affirmed on every entry
point that clones from supplied audio, and is recorded in the exported voice
file's metadata. `mynah_voice_clone_consent_notice` is one string for `--help`,
the README and the API error text.

### Two things this uncovered, both fixed

- **`model.json` did not carry the SEANet geometry** (`n_filters`,
  `kernel_size`, `residual_kernel_size`, `last_kernel_size`,
  `n_residual_layers`, `compress`), which forced both this module and E3's
  decoder to hardcode it. `tools/convert_pocket.py` now derives all six from
  tensor shapes with cross-checks. `dilation_base` is deliberately **not**
  emitted: with one residual layer only `base**0 == 1` is ever used, so the
  checkpoint cannot witness it and any value would be a guess.
- **`mynah_weights_get` could not read a voice file.** `mynah_tensor` capped
  rank at 4 and a voice KV is rank 5 `[2, 1, T, 16, 64]`, so the loader rejected
  the pack's own voices — including `alba.safetensors` from a stock pack. The
  cap is now 6. It was an explicit error rather than a silent truncation, which
  is why it was found immediately.

### Self-test is mutation-tested

Breaking three things on purpose — the pre-ELU, replicate padding changed to
zero, and a one-sample group delay — is caught by the self-test in all three
cases. A test that passes under a deliberate break is worse than no test.

## 2026-09-19 — reachable at last, and what the gap actually was

Status moves from IMPLEMENTED to **USABLE**. Nothing in `src/voice_clone.c`
changed: every number in this note still stands, and the module had been
oracle-checked to 6.4e-06 since 2026-09-12. What was missing was that **no
caller ever built a `mynah_voice_clone_config` from a real pack**, so the whole
2555-line module was unreachable from the CLI, from the server and from the
library. It was found by being asked to clone a voice for a customer report
rather than by reading the board — which said `[x] mynah-tts export-voice`, a
command that returned nothing to `grep` over the entire repository.

The glue is `mynah_engine_pocket_clone_voice` plus `--clone-voice`. It is small
because three of the four sub-configurations already exist in
`engine_pocket.c` for the decoder and the encoder mirrors them; the only value
that had never been parsed is `codec_downsample_stride`. `max_seq_len` is left
at zero in both transformers: the module sizes them from `max_seconds`, and a
number invented at the call site could only disagree.

### The verification, since "it ran" is not a result

Cloned from the pack's own `alba` voice (CC-BY-4.0, commercial use allowed) —
10.32 s of its output — then made to speak a **different** sentence, against
the same sentence from the real `alba` and from an unrelated voice as control:

| | F0 mean | F0 median | LTAS corr. vs `alba` |
|---|---|---|---|
| cloned, Neoverse-V2 / `BLAS=none` | 127.4 Hz | 121.8 | **0.9861** |
| cloned, M1 / Accelerate | 127.2 | 122.4 | 0.9839 |
| `alba` itself | 127.0 | 121.5 | — |
| a different voice (control) | 86.3 | 77.7 | 0.9222 |

Both the pitch and the long-term average spectrum track the source, and the
control separates cleanly on both. Cost: 6.73 s on the box, 6.63 s on the Mac,
for 10.32 s of reference — 129 latent frames, 130 KV positions with the BOS.

**The two platforms do not produce the same voice file.** Six of twelve tensors
match (the integer offsets); the six KV caches differ by up to 6.4% of max.
That is f16 storage plus a different BLAS plus `-ffast-math`, it is what the
project's numerical policy explicitly allows across backends, and it is written
here rather than left for someone to discover with a checksum.

### One property worth stating plainly

A cloned voice is **not a second kind of voice**. What comes out is an ordinary
pack voice file; drop it in `voices/`, add the row to `speakers.json`, and every
path that serves a predefined voice serves it — the streaming server included,
with no code anywhere aware of where it came from. The pack's own cross-check
caught the incomplete edit on the first try: `model.json` said 26 speakers while
`speakers.json` listed 27, and the loader refused.
