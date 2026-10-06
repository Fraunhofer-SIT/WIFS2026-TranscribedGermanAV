"""Cross-domain table: accuracy, F1 and AUC per masking, method and direction
(source2target), next to the in-domain value of the same method on the target
domain taken from the main results.

    python make_cross_table.py --cross results_cross/test_runs.csv --main results_main/test_runs.csv --out cross_table

Writes <out>.csv, <out>.txt and <out>.tex.
"""
import argparse
import csv
from pathlib import Path


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross", required=True)
    ap.add_argument("--main", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    indomain = {(r["topic"], r["masking"], r["model"]): r for r in read(args.main)}
    rows = []
    for r in read(args.cross):
        if "2" not in r["topic"]:
            continue
        src, tgt = r["topic"].split("2", 1)
        ref = indomain.get((tgt, r["masking"], r["model"]))
        row = {"masking": r["masking"], "model": r["model"], "source": src, "target": tgt}
        for m in ("acc", "f1", "auc"):
            row[m] = float(r[m])
            row[f"indomain_{m}"] = float(ref[m]) if ref else None
        rows.append(row)
    rows.sort(key=lambda x: (x["masking"], x["model"], x["target"], x["source"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def f(v):
        return "  n/a" if v is None else f"{v:.3f}"
    lines = [f"{'masking':<11}{'model':<32}{'train->test':<24}acc    f1     auc    | in-domain acc  f1     auc"]
    for r in rows:
        lines.append(f"{r['masking']:<11}{r['model']:<32}{r['source'] + ' -> ' + r['target']:<24}"
                     f"{r['acc']:.3f}  {r['f1']:.3f}  {r['auc']:.3f}  | "
                     f"{f(r['indomain_acc'])}          {f(r['indomain_f1'])}  {f(r['indomain_auc'])}")
    Path(f"{out}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    dirs = sorted({(r["source"], r["target"]) for r in rows}, key=lambda d: (d[1], d[0]))
    tgts = sorted({d[1] for d in dirs})
    tex = ["\\begin{tabular}{l" + "r" * (len(dirs) + len(tgts)) + "}", "\\toprule",
           "Method & " + " & ".join(f"{s}$\\rightarrow${t}" for s, t in dirs)
           + " & " + " & ".join(f"in-domain {t}" for t in tgts) + " \\\\", "\\midrule"]
    for masking in sorted({r["masking"] for r in rows}):
        tex.append("\\multicolumn{%d}{l}{\\textit{%s}} \\\\" % (1 + len(dirs) + len(tgts), masking))
        for model in sorted({r["model"] for r in rows}):
            byd = {(r["source"], r["target"]): r for r in rows if r["masking"] == masking and r["model"] == model}
            cells = [f"{byd[d]['acc']:.3f}" if d in byd else "--" for d in dirs]
            ref = []
            for t in tgts:
                d = next(x for x in dirs if x[1] == t)
                ref.append(f"{byd[d]['indomain_acc']:.3f}" if d in byd and byd[d]["indomain_acc"] is not None else "--")
            tex.append(f"{model} & " + " & ".join(cells + ref) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    Path(f"{out}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
