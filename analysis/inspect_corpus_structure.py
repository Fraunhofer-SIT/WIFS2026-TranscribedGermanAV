"""Read the structure of a structured corpus from its case folders.

    python inspect_corpus_structure.py <root> [--subdir pp2] [--masking original,posnoised]

Reports per masking and topic: case counts per split, speakers shared between
splits, the topology of the N-case graph, N-cases whose speakers belong to the
other split, and which document of a speaker serves as known and unknown in
the N-cases (matched by the video id in the file name).
"""
import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

CASE_RE = re.compile(r"^(?:\[[A-Za-z]\]\s*)?(.+?)\s+vs\.?\s+(.+?)\s*$")
ID_RE = re.compile(r" - ([A-Za-z0-9_-]{11})\.txt$")


def speakers(name):
    m = CASE_RE.match(name)
    if not m:
        return None
    return (re.sub(r"\[.*?\]", "", m.group(1)).strip(), re.sub(r"\[.*?\]", "", m.group(2)).strip())


def video_id(path):
    m = ID_RE.search(path.name)
    return m.group(1) if m else None


def components(edges, nodes):
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen, out = set(), []
    for start in sorted(nodes):
        if start in seen:
            continue
        comp, stack = set(), [start]
        while stack:
            v = stack.pop()
            if v in comp:
                continue
            comp.add(v)
            stack.extend(adj[v])
        seen |= comp
        n_edges = sum(len(adj[v]) for v in comp) // 2
        degrees = Counter(len(adj[v]) for v in comp)
        out.append((len(comp), n_edges, dict(degrees), set(degrees) == {2} and n_edges == len(comp)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--subdir", default="")
    ap.add_argument("--masking", default="original,posnoised")
    args = ap.parse_args()

    for masking in [m for m in args.masking.split(",") if m]:
        base = args.root / masking / args.subdir if args.subdir else args.root / masking
        if not base.is_dir():
            print(f"[{masking}] missing directory {base}")
            continue
        for topic_dir in sorted(d for d in base.iterdir() if d.is_dir()):
            print(f"\n== {masking}/{topic_dir.name}")
            splits = {d.name: d for d in sorted(topic_dir.iterdir()) if d.is_dir()}
            split_speakers, split_edges, y_docs, home = {}, {}, {}, {}
            for sname, sdir in splits.items():
                ys, ns, edges, spk = 0, 0, [], set()
                for case in sorted(p for p in sdir.iterdir() if p.is_dir()):
                    sp = speakers(case.name)
                    if sp is None:
                        continue
                    a, b = sp
                    spk |= {a, b}
                    if a == b:
                        ys += 1
                        home[a] = sname
                        known = sorted(case.glob("known*.txt"))
                        unknown = sorted(case.glob("unknown*.txt"))
                        if len(known) == 1 and len(unknown) == 1:
                            y_docs[a] = (video_id(known[0]), video_id(unknown[0]))
                    else:
                        ns += 1
                        edges.append((a, b))
                split_speakers[sname], split_edges[sname] = spk, edges
                print(f"  {sname}: {ys} Y, {ns} N, {len(spk)} speakers")
            names = sorted(splits)
            for i, s1 in enumerate(names):
                for s2 in names[i + 1:]:
                    shared = split_speakers[s1] & split_speakers[s2]
                    print(f"  speakers in {s1} and {s2}: {len(shared)}")
            cross = [(s, a, b) for s, es in split_edges.items() for a, b in es
                     if home.get(a, s) != s or home.get(b, s) != s]
            print(f"  N-cases with a speaker of the other split: {len(cross)}")
            all_nodes = set().union(*split_speakers.values()) if split_speakers else set()
            for sname in names:
                comps = components(split_edges[sname], split_speakers[sname] if not cross else all_nodes)
                comps = [c for c in comps if c[1] > 0]
                desc = "; ".join(f"{n} nodes/{e} edges degrees={d} {'ring' if ring else 'path'}" for n, e, d, ring in comps)
                print(f"  N-graph {sname}: {desc}")
            convention = Counter()
            for sname, edges in split_edges.items():
                for a, b in edges:
                    case = next((c for c in (splits[sname] / f"[N] {a} vs. {b}", splits[sname] / f"{a} vs. {b}") if c.is_dir()), None)
                    if case is None:
                        continue
                    known = sorted(case.glob("known*.txt"))
                    unknown = sorted(case.glob("unknown*.txt"))
                    if len(known) != 1 or len(unknown) != 1:
                        continue
                    kv, uv = video_id(known[0]), video_id(unknown[0])
                    if a in y_docs:
                        convention["known=left.doc1" if kv == y_docs[a][0] else "known=left.doc2" if kv == y_docs[a][1] else "known=left.other"] += 1
                    if b in y_docs:
                        convention["unknown=right.doc1" if uv == y_docs[b][0] else "unknown=right.doc2" if uv == y_docs[b][1] else "unknown=right.other"] += 1
            print(f"  N-case document convention: {dict(convention) or 'n/a'}")


if __name__ == "__main__":
    main()
