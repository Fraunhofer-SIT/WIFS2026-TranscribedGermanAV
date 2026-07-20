import re
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

# Pair folders are named "<known author> vs. <unknown author>" and may carry a
# label prefix, e.g. "[Y] Alice vs. Alice".
_VS_RE = re.compile(r"^(?:\[[A-Za-z]\]\s*)?(.+?)\s+vs\.?\s+(.+?)\s*$")


class AVMethod(ABC):
    """Base class for authorship verification methods.

    train() and eval() return a metrics dict as built by calc_metrics().
    """

    train_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def train(self, corpus_name: str, corpus_path: str, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def eval(self, corpus_name: str, corpus_path: str, **kwargs) -> Dict[str, Any]:
        pass


def create_truth_file(corpus_path) -> None:
    """Write truth.txt with one "<pair folder> <Y|N>" line per pair."""
    corpus_path = Path(corpus_path)
    truth_file = corpus_path / "truth.txt"
    if truth_file.exists():
        return

    entries = []
    for folder in corpus_path.iterdir():
        if not folder.is_dir():
            continue
        match = _VS_RE.match(folder.name)
        if not match:
            continue
        known = re.sub(r"\[.*?\]", "", match.group(1)).strip()
        unknown = re.sub(r"\[.*?\]", "", match.group(2)).strip()
        entries.append(f"{folder.name} {'Y' if known == unknown else 'N'}")

    if entries:
        truth_file.write_text("\n".join(entries) + "\n", encoding="utf-8")


def read_corpus(path, process_strings=None):
    """Load an AV corpus in PAN format.

    Returns the problems as (known_text, known_author, unknown_text,
    unknown_author) tuples, the labels as 0 (N) and 1 (Y), and a mapping from
    author to that author's texts.
    """
    path = Path(path)
    truth_file = path / "truth.txt"
    if not truth_file.exists():
        raise FileNotFoundError(f"no truth.txt in {path}")

    truth = {
        line[:-2]: line[-1] == "Y"
        for line in truth_file.read_text("utf-8").split("\n")
        if line
    }
    named_by_folder = all(_VS_RE.match(k) for k in truth)

    problems, labels = [], []
    author_texts = defaultdict(set)

    for problem in list(path.glob("**/"))[1:]:
        if problem.name not in truth:
            continue

        if named_by_folder:
            match = _VS_RE.match(problem.name)
            if not match:
                continue
            known_author = re.sub(r"\[.*?\]", "", match.group(1)).strip()
            unknown_author = re.sub(r"\[.*?\]", "", match.group(2)).strip()
        else:
            # Fall back to the author tag in the file names, e.g. "known [Alice].txt".
            known_files = list(problem.glob("known*"))
            unknown_files = list(problem.glob("unknown*"))
            if not known_files or not unknown_files:
                continue
            tag_k = re.findall(r"\[(\S+)", known_files[0].name)
            tag_u = re.findall(r"\[(\S+)", unknown_files[0].name)
            known_author = tag_k[0] if tag_k else problem.name
            unknown_author = tag_u[0] if tag_u else problem.name

        known_texts = [f.read_text("utf-8") for f in problem.glob("known*")]
        unknown_texts = [f.read_text("utf-8") for f in problem.glob("unknown*")]
        if process_strings:
            known_texts = [process_strings(t) for t in known_texts]
            unknown_texts = [process_strings(t) for t in unknown_texts]

        author_texts[known_author] |= set(known_texts)
        labels.append(truth[problem.name])
        problems.append(
            (" ".join(known_texts), known_author, " ".join(unknown_texts), unknown_author)
        )

    return problems, np.array(labels).astype(np.float32), author_texts


def calc_metrics(corpus_name, model_name, y_pred_scores, y_pred_labels, y_true):
    assert len(y_pred_scores) == len(y_pred_labels) == len(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels).ravel()

    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    if len(set(y_true)) > 1 and len(set(y_pred_scores)) > 1:
        auc = roc_auc_score(y_true, y_pred_scores)
    else:
        auc = 0.5

    return {
        "corpus": corpus_name,
        "model": model_name,
        "acc": acc,
        "auc": auc,
        "f1": f1,
        "prec": prec,
        "rec": rec,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "N": len(y_true),
    }
