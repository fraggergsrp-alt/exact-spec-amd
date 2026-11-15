"""BI3 summary. Usage: python bi3_summary.py 17|4b   (reads results/BI3_<m>_{base,bi}_s1.json and _s2 if present)"""
import json
import statistics
import sys
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
m = sys.argv[1]


def load(mode, s):
    p = R / f"BI3_{m}_{mode}_{s}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fd(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


sessions = [s for s in ("s1", "s2") if load("bi", s) and load("base", s)]
for s in sessions:
    b, i = load("base", s), load("bi", s)
    rb, ri = b["rows"], i["rows"]
    n = len(rb)
    idb = sum(r["fixed7"]["first_diff"] is None for r in rb)
    idi = sum(r["fixed7"]["first_diff"] is None for r in ri)
    diffs = sorted(r["fixed7"]["first_diff"] for r in rb if r["fixed7"]["first_diff"] is not None)
    fb, fi = statistics.mean(r["fixed7"]["tok_s"] for r in rb), statistics.mean(r["fixed7"]["tok_s"] for r in ri)
    ab, ai = statistics.mean(r["tok_s_target"] for r in rb), statistics.mean(r["tok_s_target"] for r in ri)
    same_plain = sum(fd(x["tokens_target"], y["tokens_target"]) is None for x, y in zip(rb, ri))
    print(f"BI3 {m} {s}: {n} prompts x {b['newt']} tokens | spec==plain normal {idb}/{n} (first diffs at {diffs}) | "
          f"BI {idi}/{n}")
    print(f"BI3 {m} {s}: fixed7 normal {fb:.1f} BI {fi:.1f} tok/s ({100 * (fi / fb - 1):+.1f}%) | target alone normal "
          f"{ab:.1f} BI {ai:.1f} ({100 * (ai / ab - 1):+.1f}%) | BI plain == normal plain on {same_plain}/{n}")
    for r in ri:
        if r["fixed7"]["first_diff"] is not None:
            print(f"BI3 {m} {s}: BI NOT IDENTICAL on '{r['prompt']}' at token {r['fixed7']['first_diff']}")
if len(sessions) == 2:
    for mode in ("base", "bi"):
        a, c = load(mode, "s1")["rows"], load(mode, "s2")["rows"]
        sp = sum(fd(x["tokens_target"], y["tokens_target"]) is None for x, y in zip(a, c))
        ss = sum(fd(x["fixed7"]["tokens"], y["fixed7"]["tokens"]) is None for x, y in zip(a, c))
        fa, fc = statistics.mean(r["fixed7"]["tok_s"] for r in a), statistics.mean(r["fixed7"]["tok_s"] for r in c)
        print(f"BI3 {m} REPRO {mode}: session 1 vs 2 same plain output {sp}/{len(a)}, same speculative output "
              f"{ss}/{len(a)} | fixed7 {fa:.1f} vs {fc:.1f} tok/s")
