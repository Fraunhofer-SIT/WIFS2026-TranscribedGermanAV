import re

FILLERS = ["\u00e4hm", "\u00e4hs", "\u00e4h", "\u00f6hm", "\u00f6h",
           "ehm", "hmm", "hm", "mhm", "mmh", "mm", "mh"]
FILLER_RE = re.compile(r"\b(?:" + "|".join(FILLERS) + r")\b", re.IGNORECASE)
ELLIPSIS_RE = re.compile(r"\u2026|(?:\.\s*){3,}")
SENT_RE = re.compile(r"(?<=[.!?])\s+")
# Latin letters including European accents, digits, whitespace and the usual
# punctuation. Everything else is dropped.
FOREIGN_RE = re.compile(
    r"[^0-9A-Za-z\u00c0-\u024f\s\.,!\?;:\-\u2013\u2014'\"\u201e\u201c\u201d()\[\]%&/+=\u20ac$@#]")


def _tidy_punct(text):
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text):
    text = ELLIPSIS_RE.sub(" ", text)
    # A missing space after punctuation helps the sentence splitter.
    text = re.sub(r"([.!?])([A-Za-z\u00c0-\u024f])", r"\1 \2", text)
    text = _tidy_punct(text)
    if text:
        text = text[0].upper() + text[1:]
    return text


def strip_foreign(text):
    text = FOREIGN_RE.sub(" ", text)
    return _tidy_punct(text)


def strip_fillers(text):
    text = FILLER_RE.sub(" ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([.!?])[,;:]+", r"\1", text)
    text = re.sub(r"(?:[,;:]\s*){2,}", ", ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\s,;:.]+", "", text).strip()
    if text:
        text = text[0].upper() + text[1:]
    return text


def _collapse_word_runs(text, max_n=10):
    toks = text.split()
    n_total = len(toks)
    out = []
    i = 0
    while i < n_total:
        matched = False
        upper = min(max_n, (n_total - i) // 2)
        for n in range(upper, 0, -1):
            gram = toks[i:i + n]
            reps = 1
            j = i + n
            while toks[j:j + n] == gram:
                reps += 1
                j += n
            threshold = 3 if n >= 2 else 5
            if reps >= threshold:
                out.extend(gram)
                i = j
                matched = True
                break
        if not matched:
            out.append(toks[i])
            i += 1
    return " ".join(out)


def collapse_repeats(text):
    text = _collapse_word_runs(text)
    # Drop a sentence identical to the one before it.
    parts = SENT_RE.split(text)
    out = []
    for p in parts:
        key = p.strip().lower()
        if out and key and key == out[-1].strip().lower():
            continue
        out.append(p)
    return " ".join(out).strip()