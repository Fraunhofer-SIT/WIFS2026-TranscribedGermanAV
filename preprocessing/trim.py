"""Mechanical cleaning and shared phrase trimming.

Removes foreign characters, ellipses and recognizer loops, then compares the two
texts of an author from both ends and cuts the greeting and closing formulas they
share. Writes copies into a new directory and leaves the input untouched.
"""

import argparse
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from clean import ELLIPSIS_RE, SENT_RE, collapse_repeats, strip_foreign

SIM_DEFAULT = 0.80
MIN_KEEP = 3


def mechanical_clean(text):
    return collapse_repeats(strip_foreign(ELLIPSIS_RE.sub(" ", text)))


def split_sentences(text):
    return [p.strip() for p in SENT_RE.split(text) if p.strip()]


def _norm(s):
    s = re.sub(r"[^0-9a-z\u00c0-\u024f\s]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _sim(a, b):
    na, nb = _norm(a), _norm(b)
    if not na and not nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def shared_prefix(sa, sb, th):
    i = 0
    while i < len(sa) and i < len(sb) and _sim(sa[i], sb[i]) >= th:
        i += 1
    return i


def shared_suffix(sa, sb, th):
    j = 0
    while j < len(sa) and j < len(sb) and _sim(sa[-1 - j], sb[-1 - j]) >= th:
        j += 1
    return j


def author_key(path):
    stem = path.stem
    if " - " not in stem:
        return None
    return (str(path.parent), stem.rsplit(" - ", 1)[0])


def trim_pair(text_a, text_b, threshold, min_keep):
    sa, sb = split_sentences(text_a), split_sentences(text_b)
    i = shared_prefix(sa, sb, threshold)
    j = shared_suffix(sa, sb, threshold)

    cap = min(len(sa), len(sb)) - min_keep
    if cap < 0:
        return text_a, text_b, 0, 0
    while i + j > cap and j > 0:
        j -= 1
    while i + j > cap and i > 0:
        i -= 1
    if i + j == 0:
        return text_a, text_b, 0, 0
    return " ".join(sa[i:len(sa) - j]), " ".join(sb[i:len(sb) - j]), i, j


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path, help="directory of prepared texts")
    parser.add_argument("--out", type=Path, default=Path("trimmed"))
    parser.add_argument("--threshold", type=float, default=SIM_DEFAULT,
                        help="similarity above which two boundary sentences count as shared")
    parser.add_argument("--min-keep", type=int, default=MIN_KEEP, dest="min_keep",
                        help="sentences that survive the trim in each text")
    args = parser.parse_args()

    files = sorted(args.src.rglob("*.txt"))
    if not files:
        raise SystemExit(f"no .txt below {args.src}")
    texts = {f: mechanical_clean(f.read_text(encoding="utf-8")) for f in files}

    groups = defaultdict(list)
    for f in files:
        key = author_key(f)
        if key:
            groups[key].append(f)

    trimmed = 0
    for (parent, author), fs in sorted(groups.items()):
        if len(fs) != 2:
            print(f"{author} in {Path(parent).name}: {len(fs)} texts, pair not trimmed")
            continue
        a, b = fs
        texts[a], texts[b], i, j = trim_pair(texts[a], texts[b], args.threshold, args.min_keep)
        if i or j:
            trimmed += 1
            print(f"{author} in {Path(parent).name}: cut {i} leading, {j} trailing")

    for f in files:
        target = args.out / f.relative_to(args.src)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(texts[f], encoding="utf-8")
    print(f"{len(files)} files written to {args.out}, {trimmed} author pairs trimmed")


if __name__ == "__main__":
    main()
