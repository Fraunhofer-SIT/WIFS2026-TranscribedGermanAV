"""Trimmed texts to verification cases.

Splits the speakers into train and test first, then within each split pairs a
speaker's two documents into a Y case and pairs adjacent speakers on a random
ring into N cases. The two splits stay author disjoint. Writes the layout run.py
expects, see CORPORA.md.
"""

import argparse
import random
import re
import shutil
from pathlib import Path

TRAIN_FRAC = 0.4
SEED = 42

# The author is everything before the trailing eleven character video id.
FILENAME_RE = re.compile(r"^(?P<author>.+) - [A-Za-z0-9_-]{11}\.txt$")


def load_text(path):
    for encoding in ("utf-8", "cp1252", "latin-1", "iso-8859-1", "utf-16"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {path}")


def collect_by_author(src):
    authors = {}
    for f in sorted(src.glob("*.txt")):
        m = FILENAME_RE.match(f.name)
        if m:
            authors.setdefault(m["author"], []).append(f)
    for a in [a for a, fs in authors.items() if len(fs) < 2]:
        del authors[a]
    if len(authors) < 2:
        raise ValueError("need at least two authors with two texts each")
    return authors


def copy_case(case_dir, known, unknown):
    case_dir.mkdir(parents=True)
    for role, src in (("known", known), ("unknown", unknown)):
        # Whitespace is normalized on the way in.
        (case_dir / f"{role} {src.name}").write_text(
            " ".join(load_text(src).split()), encoding="utf-8"
        )


def build_corpus(src, out, seed=SEED, train_frac=TRAIN_FRAC):
    authors = collect_by_author(src)
    rng = random.Random(seed)
    docs = {au: (tuple(sorted(fs)) if len(fs) == 2 else tuple(rng.sample(fs, 2)))
            for au, fs in authors.items()}

    order = sorted(authors)
    rng.shuffle(order)
    k = round(len(order) * train_frac)
    members = {"train": order[:k], "test": order[k:]}

    dst = {"train": out / "train", "test": out / "test"}
    for d in dst.values():
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    truth = {"train": [], "test": []}

    for split, speakers in members.items():
        for au in speakers:
            known, unknown = docs[au]
            case = f"[Y] {au} vs. {au}"
            copy_case(dst[split] / case, known, unknown)
            truth[split].append(f"{case} Y")

        if len(speakers) >= 2:
            ring = list(speakers)
            rng.shuffle(ring)
            for i, left in enumerate(ring):
                right = ring[(i + 1) % len(ring)]
                case = f"[N] {left} vs. {right}"
                copy_case(dst[split] / case, docs[left][0], docs[right][1])
                truth[split].append(f"{case} N")

    for split, d in dst.items():
        (d / "truth.txt").write_text("\n".join(truth[split]) + "\n", encoding="utf-8")
    return k, len(order) - k


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path, help="trimmed texts, one directory per topic")
    parser.add_argument("--out", type=Path, default=Path("corpora/original"))
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC, dest="train_frac")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    topics = [d for d in sorted(args.src.iterdir()) if d.is_dir()]
    if not topics:
        topics = [args.src]
    for topic in topics:
        try:
            n_train, n_test = build_corpus(topic, args.out / topic.name,
                                           args.seed, args.train_frac)
        except ValueError as e:
            print(f"{topic.name}: skipped, {e}")
            continue
        print(f"{topic.name}: {n_train} train speakers ({2 * n_train} cases), "
              f"{n_test} test speakers ({2 * n_test} cases)")


if __name__ == "__main__":
    main()
