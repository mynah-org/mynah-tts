# E6 — Licensing and per-voice policy in the model pack

Status: **OPEN** · small but blocking for any distribution

Not legal advice — this is a reading of the license texts, recorded so the pack
design does not have to re-derive it. Get a real opinion before shipping.

## Three licenses stacked

| Layer | License | Commercial use |
|---|---|---|
| Reference code (`kyutai-labs/pocket-tts`) | MIT | yes |
| Weights (`kyutai/pocket-tts`) | CC-BY-4.0 | yes |
| **Voices** | **per source dataset** | **not all** |

Our C runtime is unaffected: CC-BY has no share-alike, so only the weights carry
the license. Obligations on the weights are attribution, a license link, and
**stating that they were modified** — conversion and quantization are an
adaptation and must be declared.

## The part that bites: two default voices are non-commercial

Verified against `kyutai/tts-voices`, which gives a per-dataset license and
labels two of them "**Non-commercial use only.**"

| Voice | Source | License | Commercial |
|---|---|---|---|
| alba | alba-mackenna | CC BY 4.0 | yes |
| marius, javert | voice-donations | CC0 | yes |
| fantine, eponine, azelma | VCTK | CC BY 4.0 | yes |
| **jean** | **EARS** | **CC BY-NC 4.0** | **no** |
| **cosette** | **Expresso** | **CC BY-NC 4.0** | **no** |

The same 26 voices ship in every language pack, so `jean` and `cosette` are
non-commercial in the Italian, Spanish, German and Portuguese packs too.

The newer `embeddings_v3` names (anna, bill_boerst, caro_davy, charles, eve,
george, jane, mary, michael, paul, peter_yearsley, stuart_bell, vera) look like
Voice-Zero / LibriVox, which is CC0 — **but the name→file mapping is not in the
README that was read.** Verify per voice before marking any of them commercial.

## Gate terms vs license

The gated model card adds a "Prohibited use" clause (no impersonation or cloning
without lawful consent, no deception, etc.). Two separate planes:

- **CC-BY-4.0** is the license on the weights. Irrevocable, permissive, and it
  travels to whoever receives them from us.
- **"Prohibited use"** is a click-through condition of *access*. It binds whoever
  accepted it. It does **not** travel downstream, because CC-BY forbids imposing
  additional terms on recipients.

Practical position: honour it as policy anyway — the only clause with real teeth
for a TTS runtime is consent for voice cloning, which is a product flow, not a
license line — and pass it along as a **notice**, never as a license restriction.

Separate from licensing: EU AI Act art. 50 transparency duties apply to synthetic
audio in a product sold in the EU. Product-level concern, noted here so it is not
forgotten.

## Ungated mirrors

`Verylicious/pocket-tts-ungated` carries `mimi.encoder` and
`speaker_proj_weight`, i.e. the cloning-capable weights that Kyutai gated. It was
useful for reading the architecture. **It must not be the converter's source.**
Pin the official repo and record repo + revision + sha256 in `source.json`, so
the pack stays traceable if a mirror changes or disappears.

## Work items

1. `speakers.json` gains `source_dataset`, `license`, `commercial_use` per voice.
2. Default pack ships **only** CC0/CC-BY voices. `jean` and `cosette` behind an
   explicit opt-in flag, or excluded.
3. CLI and `/v1/voices` expose the license; a `--commercial-only` filter.
4. `LICENSES/MODEL_LICENSE.md` — CC-BY-4.0, attribution, and the modification
   statement ("converted to mynah pack format, quantized to int8").
5. `LICENSES/VOICES.md` — the per-voice table.
6. `NOTICE` — prohibited-use passed through as a notice.
7. Verify the `embeddings_v3` voice→dataset mapping and fill the table.
8. **The pack ships `mimi.encoder*`** — cloning is a product requirement (E7), so
   the 19.6 MB stays and the weights must come from the official **gated** Kyutai
   repo with a token, never from an ungated mirror.
9. **Consent gate for cloning.** This is the one prohibited-use clause with real
   teeth: "voice impersonation or cloning without explicit and lawful consent".
   With E7 shipped it stops being a licence footnote and becomes a product flow —
   an explicit affirmation before cloning, recorded, plus a notice in the CLI and
   the API. Design it with E7, not after.

## Acceptance gate

`mynah-tts --voices` prints a license per voice; no NC voice is in a default
pack; `source.json` points at the official repo with a pinned revision.
