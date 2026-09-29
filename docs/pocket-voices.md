# Pocket TTS voices: which one to use

The Pocket packs (`models/pocket-*`) ship the 26 built-in voices of
`kyutai/pocket-tts`. Pocket clones each voice from a ~10 s reference clip, so a
voice inherits that clip's timbre, background noise and leading silence. The
voices therefore differ in quality, and the differences come from the model's
voices, not from this runtime.

**Recommendation: use `alba`.** It is the cleanest voice we measured, it is
CC-BY-4.0 (commercial use allowed with attribution), and it is the first voice
of the pack, so it is also what the server uses when a request names no voice.

## Measured voices

Four English voices were measured on the qualification runs of
[cuda-serving.md](cuda-serving.md) (both the 6-layer and the 24-layer model,
about 13,000 transcribed clips and 1,166 clips analysed for timbre, under full
load and unloaded):

| Voice | Licence (commercial use) | WER, 24L / 6L | Metallic clips, 24L / 6L | Silence before speech, 24L / 6L | Verdict |
|---|---|---|---|---|---|
| **alba** | CC-BY-4.0 (yes) | **0.4%** / 0.5% | **0%** / 0% | 0.10 / 0.03 s | **Recommended.** Clean, clear, starts at once |
| jean | CC-BY-NC-4.0 (**no**) | 2.1% / 1.0% | 0.6% / 0% | 0.42 / 0.22 s | Clean, but non-commercial only |
| javert | CC0-1.0 (yes) | 0.7% / 0.4% | 7.6% / 3.0% | **0.73 / 1.02 s** | Intelligible; slightly rough; opens with ~1 s of silence |
| marius | CC0-1.0 (yes) | **4.8%** / 1.9% | **31%** / 2.4% | 0.00 / 0.00 s | **Avoid**, especially on the 24-layer model: rough, metallic on short sentences |

- **WER** is faster-whisper `small.en` on the audio actually streamed. It
  measures intelligibility, not sound quality: a metallic clip that reads
  correctly scores 0%.
- **Metallic** means strong broadband hiss together with smeared voice
  harmonics, measured by `tools/pocket_audio_noise.py` and confirmed by
  listening. The effect appears only on short (one-liner) and conversational
  sentences, never on medium or long ones.
- **Silence before speech** comes from the voice's reference clip. Time to
  first audio counts that silence as audio, so with javert the listener hears
  speech about a second later than the TTFA says.
- The official Python Pocket TTS engine, run without mynah-tts, shows the same
  picture: alba occasionally slightly metallic, never at marius's level.
- Load does not change any of this: the same voices and sentences behave the
  same at full load and with the card nearly idle.

## Voices not measured

The other 22 voices have not been evaluated. Mind the licence column in the
pack's `speakers.json` before using one in a product:

| Commercial use allowed | Voices |
|---|---|
| yes | alba (CC-BY-4.0), azelma, eponine, fantine (CC-BY-4.0, VCTK), javert, marius (CC0-1.0) |
| no (CC-BY-NC-4.0) | cosette (Expresso), jean (EARS) |
| unknown licence, treated as no | anna, bill_boerst, caro_davy, charles, estelle, eve, george, giovanni, jane, juergen, lola, mary, michael, paul, peter_yearsley, rafael, stuart_bell, vera |

```bash
python3 -c "import json;[print(v['name'],v['license'],v['commercial_use']) for v in json.load(open('models/pocket-en/speakers.json'))['voices']]"
```

The Italian pack ships the same voice set; none was evaluated in Italian.

## Getting the best out of a voice

- Short prompts are the weak spot. A one-liner such as "Yes, please." is where
  rough voices turn metallic; alba holds up.
- For a custom voice, the reference clip decides the result: record ~10 s of
  clean speech, no background noise or room echo, no silence at the start.
- To judge a voice on your own sentences, capture the streamed audio with
  `tools/pocket_ladder.py --save-audio`, transcribe it with
  `tools/pocket_quality.py --from-jsonl`, and run `tools/pocket_audio_noise.py`
  on the listening set. Always report quality per voice: one rough voice hides
  inside a good average.
