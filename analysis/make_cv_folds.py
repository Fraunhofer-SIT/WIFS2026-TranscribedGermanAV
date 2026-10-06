"""Build repeated in-domain speaker splits from the structured corpora.

The construction follows the corpus builder: the speakers of a domain are
shuffled, one ring over the shuffled order defines the N-cases, and the split
is a cut of that order. The Y-case of order[i] and the N-case
(order[i], order[i+1]) go to the side of index i, so one N-case per side spans
the boundary. An N-case uses the second document of the left speaker as known
and the first document of the right speaker as unknown.

Schemes:
  resplit  one independent split per seed; seed 42 with --train-frac 0.4
           reproduces the published split
  kfold    k rotations over one ring, the test block moves

Output: <dst>/<masking>/<topic>__cv<i>/{train,test}/<case>. Y-cases link to
the existing case folders, N-cases are assembled from file links. The
assignment is written to <dst>/folds.json.
"""
import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path

CASE_RE = re.compile(r"^(?:\[[A-Za-z]\]\s*)?(.+?)\s+vs\.?\s+(.+?)\s*$")
SPLIT_DIRS = {"train": ("train", "training"), "test": ("test", "testing")}


def find_split(topic_dir, kind):
    for name in SPLIT_DIRS[kind]:
        if (topic_dir / name).is_dir():
            return topic_dir / name
    return None


def harvest(topic_dir):
    """Map speaker -> (known file, unknown file) taken from the Y-cases."""
    docs = {}
    for kind in ("train", "test"):
        split_dir = find_split(topic_dir, kind)
        if split_dir is None:
            continue
        for case in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            m = CASE_RE.match(case.name)
            if not m:
                continue
            a = re.sub(r"\[.*?\]", "", m.group(1)).strip()
            b = re.sub(r"\[.*?\]", "", m.group(2)).strip()
            if a != b:
                continue
            known = sorted(case.glob("known *.txt"))
            unknown = sorted(case.glob("unknown *.txt"))
            if len(known) == 1 and len(unknown) == 1:
                if a in docs:
                    sys.exit(f"duplicate Y-case for '{a}' in {topic_dir}")
                docs[a] = (known[0], unknown[0])
    if len(docs) < 4:
        sys.exit(f"too few Y-cases in {topic_dir} ({len(docs)})")
    return docs


def ring_assignment(order, side_of):
    m = len(order)
    y = {"train": [], "test": []}
    n = {"train": [], "test": []}
    for i, speaker in enumerate(order):
        y[side_of(i)].append(speaker)
    for i in range(m):
        n[side_of(i)].append((order[i], order[(i + 1) % m]))
    return {"order": order, "y": y, "n": n}


def assignments_resplit(speakers, seeds, train_frac):
    out = []
    for seed in seeds:
        rng = random.Random(seed)
        order = sorted(speakers)
        rng.shuffle(order)
        k = round(len(order) * train_frac)
        a = ring_assignment(order, lambda i: "train" if i < k else "test")
        a["seed"] = seed
        out.append(a)
    return out


def assignments_kfold(speakers, k, seed):
    rng = random.Random(seed)
    order = sorted(speakers)
    rng.shuffle(order)
    m = len(order)
    sizes = [m // k + (1 if i < m % k else 0) for i in range(k)]
    out, pos = [], 0
    for f in range(k):
        lo, hi = pos, pos + sizes[f]
        a = ring_assignment(order, lambda i: "test" if lo <= i < hi else "train")
        a["fold"] = f + 1
        out.append(a)
        pos = hi
    return out


def put(src, dst, copy):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy:
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)
    else:
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def strip_prefix(name):
    for p in ("known ", "unknown "):
        if name.startswith(p):
            return name[len(p):]
    return name


def write_rotation(docs, assign, out_dir, copy):
    counts = {}
    for split in ("train", "test"):
        for s in assign["y"][split]:
            put(docs[s][0].parent, out_dir / split / f"[Y] {s} vs. {s}", copy)
        for left, right in assign["n"][split]:
            case_dir = out_dir / split / f"[N] {left} vs. {right}"
            left_doc2 = docs[left][1]
            right_doc1 = docs[right][0]
            put(left_doc2, case_dir / f"known {strip_prefix(left_doc2.name)}", copy)
            put(right_doc1, case_dir / f"unknown {strip_prefix(right_doc1.name)}", copy)
        counts[split] = (len(assign["y"][split]),
                         len(assign["y"][split]) + len(assign["n"][split]))
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--subdir", default="", help="level between masking and topic, e.g. pp2")
    ap.add_argument("--masking", default="original,posnoised")
    ap.add_argument("--topics", default="all")
    ap.add_argument("--scheme", choices=["resplit", "kfold"], default="resplit")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--kfold-seed", type=int, default=7)
    ap.add_argument("--train-frac", type=float, default=0.4)
    ap.add_argument("--copy", action="store_true", help="copy files instead of linking")
    args = ap.parse_args()

    maskings = [m for m in args.masking.split(",") if m]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    meta = {"scheme": args.scheme, "seeds": seeds, "k": args.k,
            "train_frac": args.train_frac, "topics": {}}

    def base_dir(masking):
        return args.src / masking / args.subdir if args.subdir else args.src / masking

    first = base_dir(maskings[0])
    if not first.is_dir():
        sys.exit(f"missing directory {first}")
    topics = sorted(d.name for d in first.iterdir() if d.is_dir())
    if args.topics != "all":
        keep = set(args.topics.split(","))
        topics = [t for t in topics if t in keep]

    for topic in topics:
        docs0 = harvest(first / topic)
        speakers = sorted(docs0)
        if args.scheme == "resplit":
            assigns = assignments_resplit(speakers, seeds, args.train_frac)
        else:
            assigns = assignments_kfold(speakers, args.k, args.kfold_seed)
        meta["topics"][topic] = {"speakers": speakers, "rotations": assigns}
        for masking in maskings:
            docs = docs0 if masking == maskings[0] else harvest(base_dir(masking) / topic)
            if sorted(docs) != speakers:
                sys.exit(f"speaker sets differ between maskings for {topic}")
            for i, assign in enumerate(assigns, start=1):
                counts = write_rotation(docs, assign, args.dst / masking / f"{topic}__cv{i}", args.copy)
                tag = f"seed {assign['seed']}" if args.scheme == "resplit" else f"fold {assign['fold']}"
                print(f"{masking}/{topic} cv{i} ({tag}): train {counts['train'][0]} speakers/"
                      f"{counts['train'][1]} cases, test {counts['test'][0]} speakers/{counts['test'][1]} cases")

    args.dst.mkdir(parents=True, exist_ok=True)
    (args.dst / "folds.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
