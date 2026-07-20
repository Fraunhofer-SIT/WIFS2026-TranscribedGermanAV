import re

from nlp import get_nlp

CAP = r"[A-Z\u00c4\u00d6\u00dc][\w\u00df]+"
INTRO_RE = re.compile(
    r"(?i:ich bin(?: der| die| euer)?|ich hei\u00dfe|mein name ist|"
    r"hier ist|hier spricht|mein name)\s+(" + CAP + r"(?:\s+" + CAP + r"){0,2})")


def intro_names(text):
    return [m.group(1).strip() for m in INTRO_RE.finditer(text)]


def ner_persons(text):
    nlp = get_nlp()
    return [e.text.strip() for e in nlp(text).ents if e.label_ == "PER"]


def channel_aliases(name, yt_author=None):
    out = set()
    for base in (name, yt_author):
        if not base:
            continue
        out.add(base.strip())
        for part in base.split(" - "):
            part = part.strip()
            if len(part) >= 4:
                out.add(part)
        for m in re.finditer(r"\(([^)]+)\)", base):
            inner = m.group(1).strip()
            if len(inner) >= 4:
                out.add(inner)
        out.add(re.sub(r"\s*\([^)]*\)", "", base).strip())
    return {a for a in out if len(a) >= 4}


def _redact(text, terms, placeholder):
    for t in sorted({x for x in terms if x and len(x) >= 2}, key=len, reverse=True):
        text = re.sub(r"\b" + re.escape(t) + r"\b", placeholder, text, flags=re.IGNORECASE)
    return text


def _expand_tokens(names):
    out = set()
    for n in names:
        out.add(n)
        for tok in n.split():
            if len(tok) >= 3:
                out.add(tok)
    return out


def detect_terms(text, name, yt_author=None, use_ner=True):
    persons = set(intro_names(text))
    if use_ner:
        try:
            persons.update(ner_persons(text))
        except OSError as exc:
            print(f"named entity recognition unavailable: {exc}")
    persons = _expand_tokens(persons)
    return channel_aliases(name, yt_author), persons


def apply_redaction(text, channels, persons):
    text = _redact(text, channels, "[KANAL]")
    text = _redact(text, persons, "[NAME]")
    return text


def anonymize(text, name, yt_author=None, use_ner=True):
    channels, persons = detect_terms(text, name, yt_author, use_ner)
    return apply_redaction(text, channels, persons)