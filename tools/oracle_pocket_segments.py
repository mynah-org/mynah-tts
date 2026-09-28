#!/usr/bin/env python3
"""Reference text segmentation for PocketTTS, for `make segment-parity`.

Offline tooling only. `src/text_segment.c` reimplements upstream PocketTTS's
`split_into_best_sentences` (pocket_tts/models/text_chunking.py, kyutai-labs/
pocket-tts, MIT): sentence boundaries first, then `, ; :` inside a sentence
that is still over `max_tokens`, then greedy packing of consecutive pieces, and
each chunk tokenized on its own. This script replays that algorithm with the
real SentencePiece model and writes, per input, the token count of every chunk
-- the quantity the engine receives as `mynah_tts_request.segment_lengths`.

The algorithm below is a port of upstream's, kept close to its structure so a
reader can diff the two. It does not import `pocket_tts` (torch is not a
dependency of this repository's tooling).

What is compared is the SEGMENTATION -- where the chunks start and end,
expressed as each chunk's token count in the original tokenization, which is
the count upstream's packer itself sums (`current_nb_of_tokens_in_chunk`). Not
upstream's re-tokenized chunk text: upstream rebuilds each chunk by decoding
its pieces and joining them with " ", which rewrites the text at every internal
sentence boundary that is not a single space ("5 p.m." becomes "5 p. m.", a
newline becomes " \n"). mynah keeps the caller's text span and tokenizes that.
Upstream's prompt preparation (capitalise, terminal period, padding) is not
applied either: mynah does not apply it to the text it tokenizes today.

Output (generated, never committed; build/ is gitignored):

    build/oracle-segments/<language>.jsonl   {"hex": "<utf-8 text>", "lengths": [...]}

Usage:
    uv run --with sentencepiece python tools/oracle_pocket_segments.py
    uv run --with sentencepiece python tools/oracle_pocket_segments.py \
        --tokenizer path/to/tokenizer.model --language english
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import sentencepiece as spm

MAX_TOKENS = 50  # upstream default_parameters.MAX_TOKEN_PER_CHUNK

EDGE_CASES = [
    "Hello world.",
    "No punctuation at all in this one so it stays a single piece of text",
    "The price rose to 3.14 dollars. Then it fell to 2.5 again!",
    "He said \"stop.\" Then he left. Did she follow? Yes!",
    "Wait... what happened here? Nobody knows.",
    "A short one. " * 12,
    ("This sentence is deliberately long, with several clauses, commas and "
     "semicolons; it keeps going: well past the fifty token limit, so the "
     "fallback split on commas, semicolons and colons has something to do, "
     "and the packer then has to regroup the pieces into chunks."),
    ("One very long sentence without any comma or semicolon or colon that goes "
     "on and on well beyond the limit so that the fallback finds nothing to "
     "split on and upstream keeps the oversized sentence whole with a warning "
     "instead of cutting it in the middle of a clause"),
    "  Leading and trailing spaces.   And a second sentence.  ",
    "Mr. Smith met Dr. Jones at 5 p.m. on Main St. and they talked.",
    "Line one.\nLine two.\n\nLine three after a blank line.",
    "Numbers like 1,000,000 and 2.5 million should stay inside their sentence.",
]


def find_tokenizer(language: str) -> str:
    pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--kyutai--pocket-tts/snapshots/*/"
        f"languages/{language}/tokenizer.model")
    hits = sorted(glob.glob(pattern))
    if not hits:
        sys.exit(f"no tokenizer.model for {language}; pass --tokenizer")
    return hits[0]


class Tok:
    def __init__(self, path: str):
        self.sp = spm.SentencePieceProcessor(model_file=path)

    def ids(self, text: str) -> list[int]:
        return self.sp.encode(text)


def _is_decimal_period_boundary(tokens, start, tok):
    prefix = tok.sp.decode(tokens[:start])
    suffix = tok.sp.decode(tokens[start:])
    return (len(prefix) >= 2 and prefix[-1] == "." and prefix[-2].isdigit()
            and bool(suffix) and suffix[0].isdigit())


def _find_boundary_indices(tokens, boundary, tok=None, skip_decimal=False):
    boundary = set(boundary)
    indices = [0]
    previous_was_boundary = False
    for idx, token in enumerate(tokens):
        if token in boundary:
            previous_was_boundary = True
        else:
            if previous_was_boundary:
                if skip_decimal and tok is not None and \
                        _is_decimal_period_boundary(tokens, idx, tok):
                    previous_was_boundary = False
                    continue
                indices.append(idx)
            previous_was_boundary = False
    indices.append(len(tokens))
    return indices


def _segments(tokens, bounds, tok):
    return [(bounds[i + 1] - bounds[i], tok.sp.decode(tokens[bounds[i]:bounds[i + 1]]))
            for i in range(len(bounds) - 1)]


def split_into_best_sentences(tok: Tok, text: str, max_tokens: int) -> list[int]:
    """Upstream's split, returning each chunk's packed token count."""
    text = text.strip()
    tokens = tok.ids(text)
    end_of_sentence = tok.ids(".!...?")[1:]
    sentences = _segments(tokens, _find_boundary_indices(tokens, end_of_sentence, tok, True),
                          tok)
    fallback = tok.ids(",;:")[1:]
    refined = []
    for n, sentence in sentences:
        if n <= max_tokens:
            refined.append((n, sentence))
            continue
        sub_tokens = tok.ids(sentence.strip())
        subs = _segments(sub_tokens, _find_boundary_indices(sub_tokens, fallback), tok)
        refined.extend(subs if len(subs) > 1 else [(n, sentence)])
    chunks, current, count = [], "", 0
    for n, sentence in refined:
        if current == "":
            current, count = sentence, n
        elif count + n > max_tokens:
            chunks.append(count)
            current, count = sentence, n
        else:
            current += " " + sentence
            count += n
    if current != "":
        chunks.append(count)
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--language", default="english")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--bank", default="tests/load_texts_en_v2.txt")
    ap.add_argument("--out", default="build/oracle-segments")
    args = ap.parse_args()

    tok = Tok(args.tokenizer or find_tokenizer(args.language))
    texts = [line.split("\t", 1)[1].rstrip("\n") for line in open(args.bank, encoding="utf-8")
             if not line.startswith("#") and "\t" in line]
    texts += EDGE_CASES
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    multi = 0
    with open(out / f"{args.language}.jsonl", "w", encoding="utf-8") as f:
        for text in texts:
            lengths = split_into_best_sentences(tok, text, MAX_TOKENS)
            lengths = [n for n in lengths if n > 0]
            multi += len(lengths) > 1
            f.write(json.dumps({"hex": text.encode("utf-8").hex(), "lengths": lengths}) + "\n")
    print(f"{out / (args.language + '.jsonl')}: {len(texts)} texts, {multi} multi-segment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
