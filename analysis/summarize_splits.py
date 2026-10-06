"""Summarize repeated splits: mean and standard deviation of accuracy, F1 and
AUC per domain, masking and method, plus the number of significantly
different split pairs and the smallest p-value from the folds_* files of
run_art_table.py.

    python summarize_splits.py --test-runs results_cv/test_runs.csv --art-dir art_out --out splits_summary

Writes <out>.csv, <out>.txt and <out>.tex.
"""
import argparse
import csv
import glob
import statistics as st
from collections import defaultdict
from pathlib import Path

ALIAS = {"dvbinde": "dv", "fevecde": "fevec", "multilingualstylerepresentation": "mlsr"}


def norm(name):
    """Map method keys of test_runs.csv and display names of predictions.csv to one key."""
    key = "".join(ch for ch in name.lower() if ch.isalnum())
    return ALIAS.get(key, key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-runs", required=True)
    ap.add_argument("--art-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    vals = defaultdict(lambda: defaultdict(list))
    with open(args.test_runs, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if "__cv" not in r["topic"]:
                continue
            k = (r["topic"].split("__cv")[0], r["masking"], r["model"])
            for metric in ("acc", "f1", "auc"):
                vals[k][metric].append(float(r[metric]))

    sig = {}
    for path in glob.glob(str(Path(args.art_dir) / "folds_*_pvalues.csv")):
        base, model, masking = Path(path).name[len("folds_"):-len("_pvalues.csv")].rsplit("_", 2)
        with open(path, newline="", encoding="utf-8") as f:
            ps = [float(r["p"]) for r in csv.DictReader(f)]
        sig[(base, norm(model), masking)] = (sum(p < args.alpha for p in ps), len(ps), min(ps) if ps else float("nan"))

    rows = []
    for (base, masking, model), m in sorted(vals.items()):
        n = len(m["acc"])
        nsig, npairs, pmin = sig.get((base, norm(model), masking), (None, None, None))
        row = {"domain": base, "masking": masking, "model": model, "n_splits": n}
        for metric in ("acc", "f1", "auc"):
            row[f"{metric}_mean"] = st.mean(m[metric])
            row[f"{metric}_sd"] = st.pstdev(m[metric]) if n > 1 else 0.0
        row.update({"sig_pairs": nsig, "n_pairs": npairs, "p_min": pmin})
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def fmt(r):
        s = "n/a" if r["sig_pairs"] is None else f"{r['sig_pairs']}/{r['n_pairs']} (min p={r['p_min']:.3f})"
        return (f"{r['domain']:<12}{r['masking']:<11}{r['model']:<32}"
                f"{r['acc_mean']:.3f}+-{r['acc_sd']:.3f}  {r['f1_mean']:.3f}+-{r['f1_sd']:.3f}  "
                f"{r['auc_mean']:.3f}+-{r['auc_sd']:.3f}  {s}")
    header = f"{'domain':<12}{'masking':<11}{'model':<32}acc           f1            auc           sig. split pairs"
    text = header + "\n" + "\n".join(fmt(r) for r in rows) + "\n"
    Path(f"{out}.txt").write_text(text, encoding="utf-8")

    tex = ["\\begin{tabular}{llrrrr}", "\\toprule", "Domain & Method & Acc. & F1 & AUC & Sig. pairs \\\\", "\\midrule"]
    for masking in sorted({r["masking"] for r in rows}):
        tex.append("\\multicolumn{6}{l}{\\textit{%s}} \\\\" % masking)
        for r in [x for x in rows if x["masking"] == masking]:
            s = "--" if r["sig_pairs"] is None else f"{r['sig_pairs']}/{r['n_pairs']}"
            tex.append(f"{r['domain']} & {r['model']} & "
                       + " & ".join(f"{r[m + '_mean']:.3f}$\\pm${r[m + '_sd']:.3f}" for m in ("acc", "f1", "auc"))
                       + f" & {s} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    Path(f"{out}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
