"""Raw transcript to corpus text: middle excerpt, anonymization, normalization,
filler removal.

Input files are named "<author> - <id>.txt". The author is read from the file
name, so no download metadata is needed. Names are detected on the full
transcript and redacted in the excerpt.
"""

import argparse
from pathlib import Path

from anonymize import apply_redaction, detect_terms
from clean import normalize_text, strip_fillers
from excerpt import middle_excerpt

TARGET_CHARS = 5000


def author_of(path):
    stem = path.stem
    return stem.rsplit(" - ", 1)[0] if " - " in stem else stem


def prepare(raw, author, target_chars=TARGET_CHARS, use_ner=True, keep_fillers=False):
    excerpt = middle_excerpt(raw, target_chars)
    channels, persons = detect_terms(raw, author, None, use_ner)
    text = normalize_text(apply_redaction(excerpt, channels, persons))
    return text if keep_fillers else strip_fillers(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path, help="directory of raw transcripts")
    parser.add_argument("--out", type=Path, default=Path("prepared"))
    parser.add_argument("--target-chars", type=int, default=TARGET_CHARS, dest="target_chars")
    parser.add_argument("--keep-fillers", action="store_true", dest="keep_fillers")
    parser.add_argument("--no-ner", action="store_true", dest="no_ner",
                        help="skip the spaCy person detection")
    args = parser.parse_args()

    files = sorted(args.src.rglob("*.txt"))
    if not files:
        raise SystemExit(f"no .txt below {args.src}")

    for path in files:
        text = prepare(
            path.read_text(encoding="utf-8"),
            author_of(path),
            args.target_chars,
            use_ner=not args.no_ner,
            keep_fillers=args.keep_fillers,
        )
        target = args.out / path.relative_to(args.src)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    print(f"{len(files)} files written to {args.out}")


if __name__ == "__main__":
    main()
