"""Run authorship verification methods over the corpora and write per-run metrics.

Expected layout below --corpora:

    <masking>/<topic>/train/
    <masking>/<topic>/test/

Each split holds one folder per verification case, named
"<known author> vs. <unknown author>", containing a known and an unknown text.
"""

import argparse
import csv
import sys
import traceback
from pathlib import Path

from common import create_truth_file
from methods import DETERMINISTIC, EXTRA, MAIN, build

CSV_COLUMNS = [
    "topic",
    "masking",
    "model",
    "run",
    "acc",
    "auc",
    "f1",
    "prec",
    "rec",
    "tp",
    "fn",
    "fp",
    "tn",
    "N",
]


def find_corpora(root, maskings, topics):
    corpora = []
    for masking in maskings:
        base = root / masking
        if not base.is_dir():
            raise SystemExit(f"no masking directory {base}")
        for topic_dir in sorted(d for d in base.iterdir() if d.is_dir()):
            if topics and topic_dir.name not in topics:
                continue
            train, test = topic_dir / "train", topic_dir / "test"
            if not train.is_dir() or not test.is_dir():
                print(f"skipping {masking}/{topic_dir.name}: no train/test split")
                continue
            for split in (train, test):
                create_truth_file(split)
            corpora.append((masking, topic_dir.name, train, test))
    if not corpora:
        raise SystemExit(f"no corpora found below {root}")
    return corpora


def save_row(row, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def free_gpu(model):
    """Release the GPU between methods, which matters on a shared card."""
    try:
        import gc

        import torch
    except ImportError:
        return
    for attr, value in list(vars(model).items()):
        if isinstance(value, torch.nn.Module):
            value.to("cpu")
            setattr(model, attr, None)
        elif torch.is_tensor(value) and value.is_cuda:
            setattr(model, attr, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_list(value, valid, what):
    if value == "all":
        return None
    names = [v.strip() for v in value.split(",") if v.strip()]
    unknown = [n for n in names if valid and n not in valid]
    if unknown:
        raise SystemExit(f"unknown {what}: {', '.join(unknown)}")
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpora", type=Path, default=Path("corpora"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--methods",
        default="all",
        help=f"comma separated; 'all' runs {', '.join(MAIN)}. Also available: {', '.join(EXTRA)}",
    )
    parser.add_argument("--masking", default="original,posnoised")
    parser.add_argument(
        "--lang",
        default="de",
        choices=["de", "en"],
        help="selects the English variant of a method where it has one",
    )
    parser.add_argument("--topics", default="all")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="fixed seed; omit so that repeated runs are independent",
    )
    args = parser.parse_args()

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
                try:
                    metrics = {
                        "train": model.train(topic, str(train)),
                        "test": model.eval(topic, str(test)),
                    }
                except Exception:
                    print(f"failed: {name} on {masking}/{topic} run {run}")
                    traceback.print_exc()
                    failed += 1
                    continue

                for split, values in metrics.items():
                    row = dict(values, topic=topic, masking=masking, model=name, run=run)
                    save_row(row, args.results / f"{split}_runs.csv")

                done += 1
                test_metrics = metrics["test"]
                print(
                    f"{name} {masking}/{topic} run {run}: "
                    f"acc={test_metrics['acc']:.3f} "
                    f"auc={test_metrics['auc']:.3f} "
                    f"f1={test_metrics['f1']:.3f}"
                )
            free_gpu(model)

    print(f"{done} of {done + failed} runs finished")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())