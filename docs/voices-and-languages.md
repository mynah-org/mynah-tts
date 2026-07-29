# Voices and languages

## Speaker IDs

`--speaker N` takes a number, and the names live in `speakers.json` inside the
model pack. For the pinned `magpie_tts_multilingual_357m` v2607 pack:

| ID | Voice |
|---|---|
| 0 | Aria |
| 1 | Jason |
| 2 | John |
| 3 | Leo |
| **4** | **Sofia** |

So the `--speaker 4` in the examples is Sofia.

These five are baked into the checkpoint. **This Magpie release has no zero-shot
voice cloning** — there is no way to add a voice from a reference clip, and the
runtime deliberately exposes no cloning API.

Read the IDs from your own pack rather than trusting this table if you convert a
different checkpoint:

```bash
cat models/magpie-v2607-pack/speakers.json
./build/cpu/mynah-tts --inspect models/magpie-v2607-pack | grep speaker_count
```

## Languages

`--lang CODE` selects the tokenizer. The v2607 pack ships 12:

| Code | Language | Tokenizer |
|---|---|---|
| `ar` | Arabic (MSA) | character |
| `de` | German | phoneme |
| `en` | English | phoneme |
| `es` | Spanish | phoneme |
| `fr` | French | character |
| `hi` | Hindi | phoneme |
| `it` | Italian | character |
| `ja` | Japanese | phoneme |
| `ko` | Korean | character |
| `pt` | Portuguese (Brazilian) | phoneme |
| `vi` | Vietnamese | character |
| `zh` | Mandarin | phoneme |

Phoneme languages go through a pronunciation dictionary and an IPA fallback;
character languages tokenize the text directly. The mapping is read from
`language_to_tokenizer` in the pack's `model.json`, not hardcoded.

```bash
python3 -c "import json;print(json.load(open('models/magpie-v2607-pack/model.json'))['languages'])"
```

## Text input

Three mutually exclusive ways to give text:

```bash
--text "hello from mynah" --lang en    # normalized and tokenized for you
--normalized 'h|ə|ˈ|l|o|ʊ'             # you supply phonemes, pipe-separated
--tokens 55,79,90,59,62                # you supply token IDs
```

`--normalized` and `--tokens` bypass text normalization entirely. Use them when
you need a reproducible input — the benchmarks and parity tests do, because
normalization is model behavior and must not drift silently between runs.

Speaker and language are independent flags. Whether a given voice sounds natural
in a given language is a property of the checkpoint, not of this runtime, and is
not something this project has evaluated across all 60 combinations.
