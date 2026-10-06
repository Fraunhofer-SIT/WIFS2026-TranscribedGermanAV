"""Drop-in replacement for run.py that also records per-case predictions.

The repository root is located by walking up from this file to run.py:

    python analysis/patched_run.py --corpora corpora_cv --results results_cv --methods all --runs 1

Arguments are those of run.py plus --predictions PATH (default
<results>/predictions.csv). Without modifying run.py or common.py, the wrapper
replaces common.read_corpus by a version with a sorted case order and wraps
common.calc_metrics so that every call appends one row per case:
topic, masking, model, run, split, pair, y_true, y_pred, score, aligned.
aligned=1 means the case names could be matched to the prediction order;
rows with aligned=0 carry index names and must not enter paired tests.
"""
import argparse
import csv
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path

_here = Path(__file__).resolve().parent
_root = next(p for p in (_here, *_here.parents) if (p / "run.py").exists())
sys.path.insert(0, str(_root))
import common  # noqa: E402
from methods import DETERMINISTIC, EXTRA, MAIN, build  # noqa: E402
from run import find_corpora, free_gpu, parse_list, save_row  # noqa: E402

CASE_RE = common._VS_RE
PRED_COLUMNS = ["topic", "masking", "model", "run", "split", "pair", "y_true", "y_pred", "score", "aligned"]

_orig_read_corpus = common.read_corpus
_orig_calc_metrics = common.calc_metrics
_ctx = {"masking": "", "run": 0, "split": "", "pred_path": None}
_read_stack = []


def read_corpus_sorted(path, process_strings=None):
    path = Path(path)
    truth_file = path / "truth.txt"
    if not truth_file.exists():
        raise FileNotFoundError(f"no truth.txt in {path}")
    truth = {line[:-2]: line[-1] == "Y" for line in truth_file.read_text("utf-8").split("\n") if line}
    named_by_folder = all(CASE_RE.match(k) for k in truth)

    problems, labels, names = [], [], []
    author_texts = defaultdict(set)
    folders = sorted((p for p in path.iterdir() if p.is_dir() and p.name in truth), key=lambda p: p.name)
    for problem in folders:
        if named_by_folder:
            m = CASE_RE.match(problem.name)
            if not m:
                continue
            known_author = re.sub(r"\[.*?\]", "", m.group(1)).strip()
            unknown_author = re.sub(r"\[.*?\]", "", m.group(2)).strip()
        else:
            known_author = problem.name + "_known"
            unknown_author = problem.name + "_unknown"
        known_texts, unknown_texts = [], []
        for f in sorted(problem.glob("*.txt")):
            text = f.read_text("utf-8", errors="ignore")
            if process_strings:
                text = process_strings(text)
            (unknown_texts if f.name.startswith("unknown") else known_texts).append(text)
        if not known_texts or not unknown_texts:
            continue
        known = "\n".join(known_texts)
        unknown = "\n".join(unknown_texts)
        problems.append((known, known_author, unknown, unknown_author))
        labels.append(1 if truth[problem.name] else 0)
        names.append(problem.name)
        author_texts[known_author].add(known)
        author_texts[unknown_author].add(unknown)
    _read_stack.append((names, list(labels)))
    return problems, labels, author_texts


def calc_metrics_dump(corpus_name, model_name, y_pred_scores, y_pred_labels, y_true):
    names, aligned = None, 0
    for cand_names, cand_labels in reversed(_read_stack):
        if list(cand_labels) == list(y_true):
            names, aligned = cand_names, 1
            break
    if names is None:
        names = [f"case{i:03d}" for i in range(len(y_true))]
    p = _ctx["pred_path"]
    if p is not None:
        write_header = not p.exists()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(PRED_COLUMNS)
            for name, yt, yp, sc in zip(names, y_true, y_pred_labels, y_pred_scores):
                w.writerow([corpus_name, _ctx["masking"], model_name, _ctx["run"], _ctx["split"],
                            name, int(yt), int(yp), float(sc), aligned])
    return _orig_calc_metrics(corpus_name, model_name, y_pred_scores, y_pred_labels, y_true)


common.read_corpus = read_corpus_sorted
common.calc_metrics = calc_metrics_dump
for mod_name, mod in list(sys.modules.items()):
    if mod is None or not mod_name.startswith("methods"):
        continue
    if getattr(mod, "read_corpus", None) is _orig_read_corpus:
        mod.read_corpus = read_corpus_sorted
    if getattr(mod, "calc_metrics", None) is _orig_calc_metrics:
        mod.calc_metrics = calc_metrics_dump


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpora", type=Path, default=Path("corpora"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--methods", default="all")
    parser.add_argument("--masking", default="original,posnoised")
    parser.add_argument("--lang", default="de", choices=["de", "en"])
    parser.add_argument("--topics", default="all")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    args = parser.parse_args()

    _ctx["pred_path"] = args.predictions or (args.results / "predictions.csv")
    names = parse_list(args.methods, MAIN + EXTRA, "method") or MAIN
    maskings = parse_list(args.masking, None, "masking") or ["original", "posnoised"]
    topics = parse_list(args.topics, None, "topic")
    corpora = find_corpora(args.corpora, maskings, topics)

    done = failed = 0
    for name in names:
        runs = 1 if name in DETERMINISTIC else args.runs
        for run in range(1, runs + 1):
            try:
                model = build(name, seed=args.seed, lang=args.lang)
            except Exception:
                print(f"failed to build {name}")
                traceback.print_exc()
                failed += len(corpora)
                break
            for masking, topic, train, test in corpora:
                _ctx["masking"], _ctx["run"] = masking, run
                _read_stack.clear()
                try:
                    _ctx["split"] = "train"
                    train_metrics = model.train(topic, str(train))
                    _ctx["split"] = "test"
                    test_metrics = model.eval(topic, str(test))
                except Exception:
                    print(f"failed: {name} on {masking}/{topic} run {run}")
                    traceback.print_exc()
                    failed += 1
                    continue
                for split, values in (("train", train_metrics), ("test", test_metrics)):
                    save_row(dict(values, topic=topic, masking=masking, model=name, run=run),
                             args.results / f"{split}_runs.csv")
                done += 1
                print(f"{name} {masking}/{topic} run {run}: acc={test_metrics['acc']:.3f} "
                      f"auc={test_metrics['auc']:.3f} f1={test_metrics['f1']:.3f}")
            free_gpu(model)
    print(f"{done} of {done + failed} runs finished")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
