#!/usr/bin/env python3
"""Dump reference token IDs from the real PocketTTS SentencePiece models.

This is offline tooling only: it never runs as part of the C runtime. It exists
because `src/sp_tokenizer.c` reimplements SentencePiece Unigram from scratch
(see `.work/tokenizer-sentencepiece.md`), and "the audio still sounds fine" is
not a test for a tokenizer. Token IDs are exact: there is no tolerance, a single
differing id is a bug, so the oracle is a plain list of (input, ids) pairs that
the C side replays and diffs, reporting the first mismatch.

Output, one file per language:

    build/oracle-tokenizer/<lang>.jsonl    {"hex": "<input in hex>", "ids": [...]}
    build/oracle-tokenizer/manifest.json   checksums, vocab sizes, counts, seed

The input is written as **hex**, not as a JSON string. Half the corpus is not
valid UTF-8 -- truncated multibyte sequences, overlongs, encoded surrogates,
lone continuation bytes -- and those are precisely the inputs that exercise the
normalizer's "invalid byte -> one U+FFFD, consuming one byte" rule. A JSON
string round trip would mangle them (and `json` refuses lone surrogates
outright), so the bytes travel as hex and the C side unhexes them.

For the same reason the corpus is fed to `sp.encode(data, out_type=int)` with
`data` being `bytes`: the `str` overload cannot express an invalid input at all,
and `bytes` is the only entry point that reaches those paths.

The corpus itself lives in `tools/corpus_pocket.py`, deterministic given --seed.
Its output is generated, never committed (CLAUDE.md rule 8); `build/` is
gitignored.

Usage:
    uv run --with sentencepiece python tools/oracle_pocket_tokenizer.py
    uv run --with sentencepiece python tools/oracle_pocket_tokenizer.py \
        --language italian --out build/oracle-tokenizer --seed 1234
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus_pocket import (  # noqa: E402  (after sys.path fix, by design)
    CORPUS_VERSION,
    LANGUAGES,
    build_corpus,
    category_counts,
)

# Where `huggingface_hub` puts the pinned snapshot. The revision is part of the
# path and there may be more than one; we resolve by glob and record the exact
# file we used, plus its sha256, in the manifest.
REPO_DIR = "models--kyutai--pocket-tts"
TOKENIZER_REL = "snapshots/*/languages/{language}/tokenizer.model"

# From `.work/tokenizer-sentencepiece.md`: 1 UNKNOWN + 3 CONTROL + 256 BYTE +
# 3740 NORMAL. A different number means a different checkpoint, and every id in
# this dump would be meaningless to the C side.
EXPECTED_VOCAB_SIZE = 4000


def default_cache() -> Path:
    """The HF hub cache, honouring HF_HUB_CACHE / HF_HOME like the library."""
    env = os.environ.get("HF_HUB_CACHE")
    if env:
        return Path(env)
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_tokenizer(cache: Path, language: str) -> Path:
    """Locate `languages/<language>/tokenizer.model` inside the hub cache.

    Fails loudly rather than falling back to the root tokenizer: the root file
    is byte-identical to english only, so a silent fallback would produce a
    German oracle from an English model.
    """
    pattern = str(cache / REPO_DIR / TOKENIZER_REL.format(language=language))
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise SystemExit(
            f"no tokenizer.model for language {language!r}\n"
            f"  looked for: {pattern}\n"
            f"  fix: hf download kyutai/pocket-tts "
            f"(or pass --hf-cache with the right cache root)"
        )
    # Several snapshots of the same repo can coexist; if they disagree we would
    # be picking one arbitrarily, so say so instead of pretending.
    digests = {sha256_file(Path(m)): m for m in matches}
    chosen = Path(matches[0])
    if len(digests) > 1:
        print(
            f"warning: {len(digests)} distinct tokenizer.model for {language}; "
            f"using {chosen}",
            file=sys.stderr,
        )
    return chosen


def dump_language(
    language: str, model_path: Path, out_dir: Path, seed: int
) -> dict:
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    vocab_size = int(sp.get_piece_size())
    if vocab_size != EXPECTED_VOCAB_SIZE:
        print(
            f"warning: {language} vocab_size={vocab_size}, "
            f"expected {EXPECTED_VOCAB_SIZE}",
            file=sys.stderr,
        )

    cases = build_corpus(language, seed)
    path = out_dir / f"{language}.jsonl"

    total_ids = 0
    max_id = -1
    with path.open("w", encoding="ascii", newline="\n") as f:
        for case in cases:
            # bytes, not str: the str overload cannot express the invalid half
            # of the corpus, which is the half worth testing.
            ids = sp.encode(case.data, out_type=int)
            total_ids += len(ids)
            if ids:
                max_id = max(max_id, max(ids))
            # separators pinned so the file is byte-stable across Python
            # versions; ensure_ascii is moot (hex + ints) but explicit.
            f.write(
                json.dumps(
                    {"hex": case.data.hex(), "ids": ids},
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
            )
            f.write("\n")

    if max_id >= vocab_size:
        raise SystemExit(
            f"{language}: emitted id {max_id} >= vocab_size {vocab_size}"
        )

    return {
        "tokenizer": str(model_path),
        "tokenizer_sha256": sha256_file(model_path),
        "vocab_size": vocab_size,
        "unk_id": int(sp.unk_id()),
        "bos_id": int(sp.bos_id()),
        "eos_id": int(sp.eos_id()),
        "pad_id": int(sp.pad_id()),
        "cases": len(cases),
        "categories": category_counts(cases),
        "input_bytes": sum(len(c.data) for c in cases),
        "total_ids": total_ids,
        "max_id": max_id,
        "jsonl": path.name,
        "jsonl_bytes": path.stat().st_size,
        "jsonl_sha256": sha256_file(path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--language",
        action="append",
        choices=LANGUAGES,
        help=f"repeatable; default: all of {', '.join(LANGUAGES)}",
    )
    ap.add_argument("--out", type=Path, default=Path("build/oracle-tokenizer"))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--hf-cache",
        type=Path,
        default=None,
        help="HuggingFace hub cache root (default: HF_HUB_CACHE / HF_HOME / ~)",
    )
    args = ap.parse_args()

    languages = args.language or list(LANGUAGES)
    cache = args.hf_cache or default_cache()
    if not cache.is_dir():
        raise SystemExit(f"HuggingFace cache not found: {cache}")

    try:
        import sentencepiece as spm
    except ImportError:
        raise SystemExit(
            "sentencepiece is not installed\n"
            "  fix: uv run --with sentencepiece python "
            "tools/oracle_pocket_tokenizer.py"
        )

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    per_language: dict[str, dict] = {}
    for language in languages:
        model_path = find_tokenizer(cache, language)
        per_language[language] = dump_language(language, model_path, out_dir, args.seed)

    manifest = {
        "generator": "tools/oracle_pocket_tokenizer.py",
        "corpus": "tools/corpus_pocket.py",
        "corpus_version": CORPUS_VERSION,
        "upstream": "kyutai/pocket-tts",
        "seed": args.seed,
        "sentencepiece": spm.__version__,
        "encode_input": "bytes",
        "record": {"hex": "input bytes, hex encoded", "ids": "sp.encode(out_type=int)"},
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "languages": per_language,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"sentencepiece {spm.__version__}  seed={args.seed}  -> {out_dir}")
    print(
        f"{'language':<12}{'cases':>8}{'ids':>10}{'vocab':>7}"
        f"{'jsonl KiB':>11}  tokenizer sha256[:12]"
    )
    for language, info in per_language.items():
        print(
            f"{language:<12}{info['cases']:>8}{info['total_ids']:>10}"
            f"{info['vocab_size']:>7}{info['jsonl_bytes'] / 1024:>11.1f}"
            f"  {info['tokenizer_sha256'][:12]}"
        )
    first = per_language[languages[0]]
    print("corpus classes: " + ", ".join(
        f"{k}={v}" for k, v in sorted(first["categories"].items())
    ))
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
