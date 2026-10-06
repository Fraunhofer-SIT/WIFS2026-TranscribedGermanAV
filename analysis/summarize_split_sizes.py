"""Split-size experiment: accuracy per method against the training fraction.

    python summarize_split_sizes.py --inputs 0.2=results_frac20/test_runs.csv 0.4=results_cv/test_runs.csv ... --out split_sizes [--seeds 3]

Each input is <train fraction>=<test_runs.csv> of a resplit run whose topics
are named <base>__cv<i>; the first --seeds repetitions per fraction are used.
Writes <out>.csv and a six-panel figure <out>.pdf/.png (domain x masking, one
line per method).
"""
import argparse
import csv
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

DOMS = [("dyi", "$\\mathcal{C}_{\\mathrm{DIY}}$"), ("finanzen", "$\\mathcal{C}_{\\mathrm{Finance}}$"),
        ("lehr-mathe", "$\\mathcal{C}_{\\mathrm{Math}}$")]
NAMES = {"coav": "COAV", "lambdag": "LambdaG", "fevec": "FeVecDiff", "stylospeaker": "StyloSpeaker",
         "mlsr": "MLSR", "rsp": "RSP", "siambert": "SiamBERT", "css": "CSS", "dv": "DV",
         "mstyledistance": "MStyleDistance"}
ORDER = list(NAMES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    vals = defaultdict(list)    # (frac, base, masking, model) -> [acc]
    sizes = {}
    for spec in args.inputs:
        frac, path = spec.split("=", 1)
        frac = float(frac)
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                m = re.match(r"(.+)__cv(\d+)$", r["topic"])
                if not m or int(m.group(2)) > args.seeds:
                    continue
                k = (frac, m.group(1), r["masking"], r["model"])
                vals[k].append(float(r["acc"]))
                sizes[frac] = int(r["N"])
    rows = []
    for (frac, base, mk, model), accs in sorted(vals.items()):
        rows.append({"train_frac": frac, "domain": base, "masking": mk, "model": model,
                     "n_splits": len(accs), "n_test": sizes[frac],
                     "acc_mean": f"{st.mean(accs):.4f}",
                     "acc_sd": f"{st.pstdev(accs) if len(accs) > 1 else 0:.4f}"})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out) + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fracs = sorted({r["train_frac"] for r in rows})
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), sharex=True, sharey=True)
    for i, (mk, mkn) in enumerate((("original", "Original"), ("posnoised", "POSNoise"))):
        for j, (d, dn) in enumerate(DOMS):
            ax = axes[i][j]
            for model in ORDER:
                pts = [(r["train_frac"], float(r["acc_mean"])) for r in rows
                       if r["domain"] == d and r["masking"] == mk and r["model"] == model]
                if not pts:
                    continue
                pts.sort()
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", markersize=3,
                        linewidth=1.1, label=NAMES.get(model, model))
            ax.set_title(f"{dn} ({mkn})", fontsize=8)
            ax.set_xticks(fracs)
            ax.set_xticklabels([f"{int(x * 100)}/{int(round((1 - x) * 100))}" for x in fracs], fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=.3)
            if i == 1:
                ax.set_xlabel("train/test speakers (%)", fontsize=7)
            if j == 0:
                ax.set_ylabel("accuracy", fontsize=7)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(str(out) + ".pdf", bbox_inches="tight")
    plt.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
