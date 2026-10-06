"""Learning curve by subsampling the training split.

build    creates corpora with n randomly drawn training cases and the
         unchanged test split: <dst>/<masking>/<topic>__lc_n<N>_d<D>/{train,test}
collect  aggregates test_runs.csv into learning_curve.csv (mean and standard
         deviation of accuracy per n) and plots one figure per masking.
"""
import argparse
import csv
import random
import re
import shutil
import statistics
import sys
import zlib
from collections import defaultdict
from pathlib import Path

LC_RE = re.compile(r"^(?P<base>.+)__lc_n(?P<n>\d+)_d(?P<d>\d+)$")


def split_dir(topic_dir, names):
    return next((topic_dir / c for c in names if (topic_dir / c).is_dir()), None)


def link(folder, dst, copy):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy:
        shutil.copytree(folder, dst)
    else:
        dst.symlink_to(folder.resolve(), target_is_directory=True)


def draw_seed(seed, masking, topic, n, d):
    return zlib.crc32(f"{seed}|{masking}|{topic}|{n}|{d}".encode("utf-8"))


def cmd_build(args):
    sizes = [int(s) for s in args.sizes.split(",")]
    for masking in [m for m in args.masking.split(",") if m]:
        base = args.src / masking / args.subdir if args.subdir else args.src / masking
        if not base.is_dir():
            sys.exit(f"missing directory {base}")
        for topic_dir in sorted(d for d in base.iterdir() if d.is_dir()):
            topic = topic_dir.name
            if args.topics != "all" and topic not in args.topics.split(","):
                continue
            tr = split_dir(topic_dir, ("train", "training"))
            te = split_dir(topic_dir, ("test", "testing"))
            if not (tr and te):
                sys.exit(f"missing train/test under {topic_dir}")
            train_cases = sorted(p for p in tr.iterdir() if p.is_dir())
            test_cases = sorted(p for p in te.iterdir() if p.is_dir())
            for n in sizes:
                if n > len(train_cases):
                    print(f"{masking}/{topic}: n={n} exceeds {len(train_cases)} training cases, skipped")
                    continue
                for d in range(1, args.draws + 1):
                    rng = random.Random(draw_seed(args.seed, masking, topic, n, d))
                    name = f"{topic}__lc_n{n:02d}_d{d:02d}"
                    for folder in rng.sample(train_cases, n):
                        link(folder, args.dst / masking / name / "train" / folder.name, args.copy)
                    for folder in test_cases:
                        link(folder, args.dst / masking / name / "test" / folder.name, args.copy)
            print(f"{masking}/{topic}: sizes {sizes}, {args.draws} draws")


def cmd_collect(args):
    acc = defaultdict(list)
    with open(args.results, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = LC_RE.match(row["topic"])
            if m:
                acc[(m.group("base"), row["masking"], row["model"], int(m.group("n")))].append(float(row["acc"]))

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "learning_curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["topic", "masking", "model", "n_train_pairs", "draws", "acc_mean", "acc_std", "acc_min", "acc_max"])
        for (base, masking, model, n), vals in sorted(acc.items()):
            w.writerow([base, masking, model, n, len(vals), f"{statistics.mean(vals):.4f}",
                        f"{statistics.pstdev(vals):.4f}", f"{min(vals):.4f}", f"{max(vals):.4f}"])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for masking in sorted({k[1] for k in acc}):
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for (base, mk, model) in sorted({k[:3] for k in acc if k[1] == masking}):
            ns = sorted(n for (b, m2, mo, n) in acc if (b, m2, mo) == (base, mk, model))
            means = [statistics.mean(acc[(base, mk, model, n)]) for n in ns]
            stds = [statistics.pstdev(acc[(base, mk, model, n)]) for n in ns]
            ax.errorbar(ns, means, yerr=stds, marker="o", capsize=3, label=base)
        ax.set_xlabel("training cases")
        ax.set_ylabel("accuracy")
        ax.set_title(masking)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out / f"learning_curve_{masking}.png", dpi=200)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--src", type=Path, required=True)
    b.add_argument("--dst", type=Path, required=True)
    b.add_argument("--masking", default="original,posnoised")
    b.add_argument("--topics", default="all")
    b.add_argument("--subdir", default="")
    b.add_argument("--sizes", default="5,10,20,30,40")
    b.add_argument("--draws", type=int, default=10)
    b.add_argument("--seed", type=int, default=11)
    b.add_argument("--copy", action="store_true")
    b.set_defaults(func=cmd_build)
    c = sub.add_parser("collect")
    c.add_argument("--results", type=Path, required=True, help="path to test_runs.csv")
    c.add_argument("--out", type=Path, default=Path("lc_out"))
    c.set_defaults(func=cmd_collect)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
