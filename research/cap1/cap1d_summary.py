"""CAP1d summary: robustness (bi3 50 x 512), repeat of the b6 result, llama.cpp Q8_0 reference, session 1 vs 2."""
import json
import statistics
from pathlib import Path

R = Path(__file__).resolve().parents[2] / "results"


def load(n):
    p = R / f"{n}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fd(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


for s in ("s1", "s2"):
    for ps in ("b6", "bi3"):
        d = load(f"CAP1d_{ps}_{s}")
        if not d:
            continue
        rs = d["rows"]
        p = statistics.mean(r["plain_tok_s"] for r in rs)
        q = statistics.mean(r["spec_tok_s"] for r in rs)
        ident = sum(r["spec_vs_plain_first_diff"] is None for r in rs)
        print(f"CAP1d {s} ours {ps:<3} ({len(rs)} x {d['newt']}): plain {p:5.1f} | exact spec {q:5.1f} tok/s "
              f"({q / p:.2f}x) | spec==plain {ident}/{len(rs)} | slower than plain on {sum(r['spec_tok_s'] < r['plain_tok_s'] for r in rs)}")
    for name in (f"B6_llamacpp_lemonade_8b_q8_{s}",):
        d = load(name)
        if d:
            for cfg in ("none", "draft7"):
                rs = [r for r in d["rows"] if r["config"] == cfg]
                if rs:
                    print(f"CAP1d {s} llama.cpp Q8_0 (lossy, fits) {cfg:<6} (11 x 96): {statistics.mean(r['tok_s'] for r in rs):5.1f} tok/s")
a, b = load("CAP1d_bi3_s1"), load("CAP1d_bi3_s2")
if a and b:
    sp = sum(fd(x["tokens_plain"], y["tokens_plain"]) is None for x, y in zip(a["rows"], b["rows"]))
    ss = sum(fd(x["tokens_spec"], y["tokens_spec"]) is None for x, y in zip(a["rows"], b["rows"]))
    print(f"CAP1d REPRO bi3 session 1 vs 2: same plain output {sp}/{len(a['rows'])} | same speculative output {ss}/{len(a['rows'])}")
