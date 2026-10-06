"""Approximate randomization tests on per-case predictions written by patched_run.py.

The test itself is the unmodified logic of art3/art.py (Van Asch, v3.0.3,
ported to Python 3). Paired comparisons on one test set use art.main(),
comparisons between different test sets use art.main3() on per-case
correctness with stratified shuffling. Significance is marked as in the PAN
2014 overview: *** p < 0.001, ** p < 0.01, * p < 0.05, = otherwise.

Modes:
  methods  all methods on one corpus, paired
           --corpus dyi --masking posnoised
  sources  training sources for one target domain, paired
           --target finanzen --model COAV --masking posnoised [--include-indomain]
  folds    repeated splits <base>__cv* of one method, stratified
           --base dyi --model COAV --masking posnoised
  sizes    training fractions of one method, stratified by split index
           --base dyi --model COAV --masking posnoised
           --inputs 0.2=results_frac20/predictions.csv 0.4=results_cv/predictions.csv ...

Outputs per call: <out>/<tag>_pvalues.csv, _table.txt and _table.tex.
"""
import argparse
import contextlib
import csv
import io
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "art3"))
try:
    import art  # noqa: E402
except ImportError:
    sys.exit("art3 is missing, run analysis/port_art.py first")

CV_RE = re.compile(r"^(?P<base>.+)__cv(?P<i>\d+)$")


def stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "="


def load_predictions(paths, split, run):
    """(topic, masking, model) -> {pair: (y_true, y_pred, aligned)}"""
    data = defaultdict(dict)
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["split"] != split or (run is not None and int(r["run"]) != run):
                    continue
                data[(r["topic"], r["masking"], r["model"])][r["pair"]] = (
                    int(r["y_true"]), int(r["y_pred"]), int(r.get("aligned", 1)))
    return data


def paired_p(gold, s1, s2, r, exact_threshold):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        probs = art.main(gold, s1, s2, N=r, exact_threshold=exact_threshold, verbose=False)
    return probs["accuracy"]


def strata_p(strata, r):
    """strata: {name: (values_a, values_b)}"""
    data = {s: {"g1": [float(v) for v in a], "g2": [float(v) for v in b]} for s, (a, b) in strata.items()}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        probs = art.main3(data, N=r, verbose=False)
    return probs["mean"]


def aligned_pairs(d1, d2):
    gold, s1, s2 = [], [], []
    for n in sorted(set(d1) & set(d2)):
        yt1, yp1, a1 = d1[n]
        yt2, yp2, a2 = d2[n]
        if not (a1 and a2):
            continue
        if yt1 != yt2:
            sys.exit(f"gold label conflict for pair '{n}'")
        gold.append("Y" if yt1 else "N")
        s1.append("Y" if yp1 else "N")
        s2.append("Y" if yp2 else "N")
    return gold, s1, s2


def correctness(d):
    return [1 if yt == yp else 0 for yt, yp, _ in d.values()]


def accuracy(d):
    c = correctness(d)
    return sum(c) / len(c) if c else float("nan")


def pan_matrix_tex(labels, accs, P, caption="", label_fmt=None, highlight_first=False):
    """Upper triangular matrix in the style of the PAN 2014 overview."""
    order = sorted(labels, key=lambda l: -accs[l])
    show = label_fmt or (lambda l: l)
    tex = ["% " + caption if caption else "%",
           "\\begin{tabular}{l" + "c" * len(order) + "}", "\\toprule",
           " & " + " & ".join("\\rotatebox{90}{" + show(l) + "}" for l in order) + " \\\\", "\\midrule"]
    for i, a in enumerate(order):
        row = [f"{show(a)} ({accs[a]:.3f})"]
        for j, b in enumerate(order):
            if j < i:
                row.append("")
            elif i == j:
                row.append("-")
            else:
                row.append(stars(P.get((a, b), P.get((b, a)))))
        line = " & ".join(row) + " \\\\"
        if highlight_first and i == 0:
            line = "\\rowcolor{yellow!25} " + line
        tex.append(line)
    tex += ["\\bottomrule", "\\end{tabular}"]
    return tex


def write_tables(labels, accs, P, out, tag, r, caption):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{tag}_pvalues.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "acc_a", "acc_b", "p", "stars"])
        for (a, b), p in sorted(P.items()):
            w.writerow([a, b, f"{accs[a]:.4f}", f"{accs[b]:.4f}", f"{p:.6g}", stars(p)])
    width = max(len(l) for l in labels) + 2
    lines = [caption, f"approximate randomization test, r={r}, *** p<0.001, ** p<0.01, * p<0.05, = otherwise", "",
             " " * width + "".join(f"{l:>{width}}" for l in labels)]
    for a in labels:
        cells = [f"{'-':>{width}}" if a == b else f"{stars(P.get((a, b), P.get((b, a)))):>{width}}" for b in labels]
        lines.append(f"{a:<{width}}" + "".join(cells))
    lines += ["", "accuracies: " + ", ".join(f"{l}={accs[l]:.3f}" for l in labels)]
    (out / f"{tag}_table.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / f"{tag}_table.tex").write_text("\n".join(pan_matrix_tex(labels, accs, P, caption)) + "\n", encoding="utf-8")
    print("\n".join(lines))


def dump_paired_inputs(out, tag, a, b, gold, s1, s2):
    d = out / "art_inputs"
    d.mkdir(parents=True, exist_ok=True)
    for ext, seq in (("gold", gold), ("s1", s1), ("s2", s2)):
        (d / f"{tag}__{a}_VS_{b}.{ext}").write_text("\n".join(seq) + "\n")


def mode_methods(args, data):
    sel = {m: d for (t, mk, m), d in data.items() if t == args.corpus and mk == args.masking}
    labels = sorted(sel)
    if len(labels) < 2:
        sys.exit(f"fewer than two methods for {args.corpus}/{args.masking}")
    accs = {l: accuracy(sel[l]) for l in labels}
    P, n = {}, 0
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            gold, s1, s2 = aligned_pairs(sel[a], sel[b])
            n = len(gold)
            P[(a, b)] = paired_p(gold, s1, s2, args.r, args.exact_threshold)
            if args.dump_inputs:
                dump_paired_inputs(args.out, f"{args.corpus}_{args.masking}", a, b, gold, s1, s2)
    write_tables(labels, accs, P, args.out, f"methods_{args.corpus}_{args.masking}", args.r,
                 f"methods on {args.corpus}/{args.masking}, {args.split} split, n={n}")


def mode_sources(args, data):
    sel = {}
    suffix = f"2{args.target}"
    for (t, mk, m), d in data.items():
        if mk != args.masking or m != args.model:
            continue
        if t.endswith(suffix):
            sel[t[:-len(suffix)]] = d
        elif args.include_indomain and t == args.target:
            sel["indomain"] = d
    labels = sorted(sel)
    if len(labels) < 2:
        sys.exit(f"fewer than two sources for target {args.target}")
    accs = {l: accuracy(sel[l]) for l in labels}
    P, n = {}, 0
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            gold, s1, s2 = aligned_pairs(sel[a], sel[b])
            n = len(gold)
            P[(a, b)] = paired_p(gold, s1, s2, args.r, args.exact_threshold)
            if args.dump_inputs:
                dump_paired_inputs(args.out, f"to_{args.target}_{args.model}_{args.masking}", a, b, gold, s1, s2)
    write_tables(labels, accs, P, args.out, f"sources_to_{args.target}_{args.model}_{args.masking}", args.r,
                 f"training sources for target {args.target} ({args.model}, {args.masking}), common cases n={n}")


def mode_folds(args, data):
    groups = {}
    for (t, mk, m), d in data.items():
        if mk != args.masking or m != args.model:
            continue
        cm = CV_RE.match(t)
        if cm and cm.group("base") == args.base:
            groups[f"cv{cm.group('i')}"] = correctness(d)
        elif t == args.base:
            groups["orig"] = correctness(d)
    labels = sorted(groups)
    if len(labels) < 2:
        sys.exit(f"fewer than two splits for {args.base}/{args.masking}")
    accs = {g: sum(v) / len(v) for g, v in groups.items()}
    P = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            P[(a, b)] = strata_p({f"{args.base}_{args.masking}": (groups[a], groups[b])}, args.r)
    write_tables(labels, accs, P, args.out, f"folds_{args.base}_{args.model}_{args.masking}", args.r,
                 f"repeated splits of {args.base} ({args.model}, {args.masking}), stratified test on per-case correctness")


def mode_sizes(args):
    """Groups are training fractions; strata are the split indices shared by all fractions."""
    groups = defaultdict(dict)
    for spec in args.inputs:
        frac, path = spec.split("=", 1)
        data = load_predictions([path], args.split, args.run)
        for (t, mk, m), d in data.items():
            cm = CV_RE.match(t)
            if mk == args.masking and m == args.model and cm and cm.group("base") == args.base:
                groups[frac][int(cm.group("i"))] = correctness(d)
    labels = sorted(groups, key=float)
    if len(labels) < 2:
        sys.exit(f"fewer than two fractions for {args.base}/{args.masking}")
    shared = set.intersection(*(set(groups[g]) for g in labels))
    shared = sorted(shared)[:args.splits] if args.splits else sorted(shared)
    if not shared:
        sys.exit("no split index shared by all fractions")
    accs = {g: sum(sum(groups[g][i]) for i in shared) / sum(len(groups[g][i]) for i in shared) for g in labels}
    P = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            P[(a, b)] = strata_p({f"cv{s}": (groups[a][s], groups[b][s]) for s in shared}, args.r)
    write_tables(labels, accs, P, args.out, f"sizes_{args.base}_{args.model}_{args.masking}", args.r,
                 f"training fractions on {args.base} ({args.model}, {args.masking}), splits {shared}, stratified by split")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictions", type=Path, nargs="*", default=[])
    ap.add_argument("--extra-predictions", type=Path, nargs="*", default=[])
    ap.add_argument("--inputs", nargs="*", default=[], help="sizes: <fraction>=<predictions.csv>")
    ap.add_argument("--mode", choices=["methods", "sources", "folds", "sizes"], required=True)
    ap.add_argument("--masking", required=True)
    ap.add_argument("--corpus")
    ap.add_argument("--target")
    ap.add_argument("--base")
    ap.add_argument("--model")
    ap.add_argument("--include-indomain", action="store_true")
    ap.add_argument("--splits", type=int, default=0, help="sizes: number of split indices to use")
    ap.add_argument("--split", default="test")
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--r", type=int, default=1000)
    ap.add_argument("--exact-threshold", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("art_out"))
    ap.add_argument("--dump-inputs", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    if args.mode == "sizes":
        if not (args.inputs and args.base and args.model):
            sys.exit("sizes requires --inputs, --base and --model")
        mode_sizes(args)
        return
    if not args.predictions:
        sys.exit("--predictions is required")
    data = load_predictions(list(args.predictions) + list(args.extra_predictions), args.split, args.run)
    if args.mode == "methods":
        if not args.corpus:
            sys.exit("--corpus is required")
        mode_methods(args, data)
    elif args.mode == "sources":
        if not (args.target and args.model):
            sys.exit("--target and --model are required")
        mode_sources(args, data)
    else:
        if not (args.base and args.model):
            sys.exit("--base and --model are required")
        mode_folds(args, data)


if __name__ == "__main__":
    main()
