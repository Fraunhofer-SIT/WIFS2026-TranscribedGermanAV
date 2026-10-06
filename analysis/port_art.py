"""Fetch the approximate randomization test of Van Asch and port it to Python 3.

    python analysis/port_art.py [--dest analysis/art3] [--src DIR]

Downloads art.py, confusionmatrix.py and combinations.py from the ruzicka
repository (or copies them from --src), verifies their checksums and applies
art3.patch, which contains the mechanical Python 3 changes listed in the patch
header. The ported files are written to --dest and are not part of this
repository.
"""
import argparse
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/mikekestemont/ruzicka/master/code/ruzicka/"
FILES = {
    "art.py": "aee802f87ee4359b38ae9e6b32b6006dc8f92b5fa6fcf698f966553e874d319b",
    "confusionmatrix.py": "98ae55742532363827a78bd8240d18010839f30209aaec308e6a23e54b7146c6",
    "combinations.py": "79c82a9ebccf0e6a1d390c2bd7929e0ca48c6ad01294cfd68c55fadacd7fbc1f",
}
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def fetch(name, src):
    if src:
        data = (Path(src) / name).read_bytes()
    else:
        with urllib.request.urlopen(BASE + name) as r:
            data = r.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != FILES[name]:
        sys.exit(f"checksum mismatch for {name}: {digest}")
    return data.decode("utf-8").split("\n")


def parse_patch(text):
    """{file: [(old_start, old_len, new_lines_without_prefix, old_lines)]}"""
    patches, current = {}, None
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- a/"):
            current = line[len("--- a/"):]
            patches[current] = []
            i += 2
            continue
        m = HUNK.match(line)
        if m:
            old_start = int(m.group(1))
            old_len = int(m.group(2)) if m.group(2) is not None else 1
            old, new = [], []
            i += 1
            while i < len(lines) and lines[i][:1] in ("-", "+", " ") and not lines[i].startswith("--- a/"):
                if lines[i][0] == "-":
                    old.append(lines[i][1:])
                elif lines[i][0] == "+":
                    new.append(lines[i][1:])
                else:
                    old.append(lines[i][1:])
                    new.append(lines[i][1:])
                i += 1
            patches[current].append((old_start, old_len, old, new))
            continue
        i += 1
    return patches


def apply(lines, hunks):
    out, pos = [], 0
    for old_start, old_len, old, new in hunks:
        start = old_start - 1 if old_len else old_start
        out.extend(lines[pos:start])
        if lines[start:start + old_len] != old:
            sys.exit(f"patch context mismatch at line {old_start}")
        out.extend(new)
        pos = start + old_len
    out.extend(lines[pos:])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, default=Path(__file__).resolve().parent / "art3")
    ap.add_argument("--src", help="directory with the original files instead of downloading")
    args = ap.parse_args()
    patches = parse_patch((Path(__file__).resolve().parent / "art3.patch").read_text(encoding="utf-8"))
    args.dest.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        ported = apply(fetch(name, args.src), patches.get(name, []))
        (args.dest / name).write_text("\n".join(ported), encoding="utf-8")
    (args.dest / "__init__.py").touch()
    print(f"ported files written to {args.dest}")


if __name__ == "__main__":
    main()
