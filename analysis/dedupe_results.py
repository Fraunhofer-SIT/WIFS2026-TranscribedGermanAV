"""Remove duplicate rows from results directories written by concurrent runs.

    python dedupe_results.py results_main [results_cv ...]

Per key (topic, masking, model, run) the last row of test_runs.csv and
train_runs.csv is kept. In predictions.csv the rows of one calc_metrics call
form a contiguous block; per key and split the block whose accuracy matches
the kept runs row is retained, otherwise the last block. Originals are saved
as *.csv.bak when something is removed.
"""
import csv
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def write(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def dedupe_runs(path):
    if not path.exists():
        return {}
    fields, rows = read(path)
    keep = OrderedDict()
    for r in rows:
        keep[(r["topic"], r["masking"], r["model"], r["run"])] = r
    removed = len(rows) - len(keep)
    if removed:
        path.rename(path.with_suffix(".csv.bak"))
        write(path, fields, list(keep.values()))
    print(f"  {path.name}: {len(rows)} -> {len(keep)} rows ({removed} removed)")
    return {k: float(v["acc"]) for k, v in keep.items()}


def dedupe_predictions(path, acc_ref):
    if not path.exists():
        return
    fields, rows = read(path)
    blocks = []
    for r in rows:
        k = (r["topic"], r["masking"], r["model"], r["run"], r["split"])
        if blocks and blocks[-1][0] == k:
            blocks[-1][1].append(r)
        else:
            blocks.append((k, [r]))
    per_key = defaultdict(list)
    for k, b in blocks:
        per_key[k].append(b)
    kept, removed = [], 0
    for k, cands in per_key.items():
        chosen = cands[-1]
        if len(cands) > 1:
            ref = acc_ref.get(k)
            if ref is not None:
                match = [b for b in cands
                         if abs(sum(int(x["y_true"]) == int(x["y_pred"]) for x in b) / len(b) - ref) < 5e-4]
                if match:
                    chosen = match[-1]
            removed += sum(len(b) for b in cands) - len(chosen)
        kept.append((k, chosen))
    if removed:
        path.rename(path.with_suffix(".csv.bak"))
        write(path, fields, [r for _, b in kept for r in b])
    print(f"  {path.name}: {len(rows)} -> {len(rows) - removed} rows ({removed} removed)")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for d in map(Path, sys.argv[1:]):
        print(f"== {d}")
        ref = {}
        for split, name in (("test", "test_runs.csv"), ("train", "train_runs.csv")):
            for k, v in dedupe_runs(d / name).items():
                ref[k + (split,)] = v
        dedupe_predictions(d / "predictions.csv", ref)


if __name__ == "__main__":
    main()
