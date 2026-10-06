"""Build cross-domain corpora in the run.py layout.

For every ordered pair of domains (S, T) with S != T:
    <dst>/<masking>/<S>2<T>/train   cases of the source domain S
    <dst>/<masking>/<S>2<T>/test    cases of the target domain T

--train-scope full|train and --test-scope full|test select whether both
original splits of a domain are merged or only the respective split is used.
"""
import argparse
import shutil
import sys
from itertools import permutations
from pathlib import Path

SPLIT_DIRS = {"train": ("train", "training"), "test": ("test", "testing")}


def find_split(topic_dir, kind):
    for name in SPLIT_DIRS[kind]:
        if (topic_dir / name).is_dir():
            return topic_dir / name
    sys.exit(f"missing directory {topic_dir}/{kind}")


def cases_of(topic_dir, scope):
    splits = ("train", "test") if scope == "full" else (scope,)
    out, seen = [], set()
    for split in splits:
        for folder in sorted(p for p in find_split(topic_dir, split).iterdir() if p.is_dir()):
            if folder.name in seen:
                sys.exit(f"case '{folder.name}' occurs in both splits of {topic_dir}")
            seen.add(folder.name)
            out.append(folder)
    return out


def link(folder, dst, copy):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy:
        shutil.copytree(folder, dst)
    else:
        dst.symlink_to(folder.resolve(), target_is_directory=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--masking", default="original,posnoised")
    ap.add_argument("--topics", default="all")
    ap.add_argument("--subdir", default="", help="level between masking and topic, e.g. pp2")
    ap.add_argument("--train-scope", choices=["full", "train"], default="full")
    ap.add_argument("--test-scope", choices=["full", "test"], default="full")
    ap.add_argument("--copy", action="store_true", help="copy files instead of linking")
    args = ap.parse_args()

    for masking in [m for m in args.masking.split(",") if m]:
        base = args.src / masking / args.subdir if args.subdir else args.src / masking
        if not base.is_dir():
            sys.exit(f"missing directory {base}")
        topics = sorted(d.name for d in base.iterdir() if d.is_dir())
        if args.topics != "all":
            keep = set(t for t in args.topics.split(",") if t)
            topics = [t for t in topics if t in keep]
        for s, t in permutations(topics, 2):
            name = f"{s}2{t}"
            for folder in cases_of(base / s, args.train_scope):
                link(folder, args.dst / masking / name / "train" / folder.name, args.copy)
            for folder in cases_of(base / t, args.test_scope):
                link(folder, args.dst / masking / name / "test" / folder.name, args.copy)
            n_tr = len(list((args.dst / masking / name / "train").iterdir()))
            n_te = len(list((args.dst / masking / name / "test").iterdir()))
            print(f"{masking}/{name}: train={n_tr} ({args.train_scope}) test={n_te} ({args.test_scope})")


if __name__ == "__main__":
    main()
