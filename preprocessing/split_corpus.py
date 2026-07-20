"""Split a corpus into train and test.

Author disjoint by default, so both texts of an author land on the same side, and
per topic directory. Copies only, the input stays untouched.
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

TEST_FRAC = 0.60
SEED = 42


def author_of(path):
    stem = path.stem
    return stem.rsplit(" - ", 1)[0] if " - " in stem else stem


def split_group(files, by, test_frac, rng):
    if by == "text":
        fs = sorted(files)
        rng.shuffle(fs)
        n_test = round(len(fs) * test_frac)
        return fs[n_test:], fs[:n_test], len(fs), n_test

    by_author = defaultdict(list)
    for f in files:
        by_author[author_of(f)].append(f)
    authors = sorted(by_author)
    rng.shuffle(authors)
    n_test = round(len(authors) * test_frac)
    test = [f for a in authors[:n_test] for f in by_author[a]]
    train = [f for a in authors[n_test:] for f in by_author[a]]
    return train, test, len(authors), n_test


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path, help="corpus with one directory per topic")
    parser.add_argument("--out", type=Path, default=Path("split"))
    parser.add_argument("--test-frac", type=float, default=TEST_FRAC, dest="test_frac")
    parser.add_argument("--by", choices=["author", "text"], default="author")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--flat", action="store_true", help="no topic directories")
    args = parser.parse_args()

    if not 0.0 < args.test_frac < 1.0:
        raise SystemExit("--test-frac must lie between 0 and 1")

    rng = random.Random(args.seed)
    subdirs = [d for d in sorted(args.src.iterdir()) if d.is_dir()]
    if args.flat or not subdirs:
        groups = {".": sorted(args.src.rglob("*.txt"))}
    else:
        groups = {d.name: sorted(d.rglob("*.txt")) for d in subdirs}

    for name, files in groups.items():
        if not files:
            continue
        train, test, n_units, n_test = split_group(files, args.by, args.test_frac, rng)
        unit = "authors" if args.by == "author" else "texts"
        print(f"{name}: {n_units} {unit}, {n_test} to test. "
              f"Files: train {len(train)}, test {len(test)}")
        for split, flist in (("train", train), ("test", test)):
            dest = args.out / name / split if name != "." else args.out / split
            dest.mkdir(parents=True, exist_ok=True)
            for f in flist:
                shutil.copy2(f, dest / f.name)


if __name__ == "__main__":
    main()
