# E7 — Voice cloning

Status: **OPEN** · **required product feature** · needs E3

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
