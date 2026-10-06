"""Bootstrap confidence intervals of accuracy over test cases.

    python bootstrap_ci.py --predictions results_main/predictions.csv --out bootstrap_ci.csv [--B 1000] [--alpha 0.05]

Per (topic, masking, model, run) the test cases are resampled with
replacement B times; the percentile interval [alpha/2, 1-alpha/2] is reported.
Columns: topic, masking, model, run, n, acc, lo, hi.
"""
import argparse
import csv
import random
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    corr = defaultdict(list)
    with open(args.predictions, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] != args.split:
                continue
            k = (r["topic"], r["masking"], r["model"], r["run"])
            corr[k].append(int(int(r["y_true"]) == int(r["y_pred"])))

    rng = random.Random(args.seed)
    rows = []
    for k in sorted(corr):
        c = corr[k]
        n = len(c)
        accs = sorted(sum(rng.choice(c) for _ in range(n)) / n for _ in range(args.B))
        lo = accs[int(args.alpha / 2 * args.B)]
        hi = accs[min(args.B - 1, int((1 - args.alpha / 2) * args.B))]
        rows.append({"topic": k[0], "masking": k[1], "model": k[2], "run": k[3], "n": n,
                     "acc": f"{sum(c) / n:.4f}", "lo": f"{lo:.4f}", "hi": f"{hi:.4f}"})
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
