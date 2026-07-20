from nlp import get_nlp


def split_sentences(text):
    doc = get_nlp()(text)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    return sents if sents else [text]


def middle_excerpt(text, target_chars):
    if len(text) <= target_chars:
        return text
    sents = split_sentences(text)
    if len(sents) <= 1:
        return text
    starts, pos = [], 0
    for s in sents:
        starts.append(pos)
        pos += len(s) + 1
    mid = len(text) // 2
    center = 0
    for i, st in enumerate(starts):
        if st <= mid:
            center = i
        else:
            break
    lo = hi = center
    total = len(sents[center])
    extend_hi = True
    while total < target_chars and (lo > 0 or hi < len(sents) - 1):
        if extend_hi and hi < len(sents) - 1:
            hi += 1
            total += len(sents[hi]) + 1
        elif lo > 0:
            lo -= 1
            total += len(sents[lo]) + 1
        extend_hi = not extend_hi
    return " ".join(sents[lo:hi + 1]).strip()