"""B6: the strongest baseline - current official llama.cpp on the same GPU, model, prompts and greedy decoding, with
each of its speculative modes. (pre-registered: docs/PREREG_B6.md)

For every configuration: start llama-server, send the 11 shared prompts (b6_prompts.py) as token ids, 96 new tokens,
temperature 0, end-of-text ignored, prompt cache off; 1 warm-up + 3 measured requests per prompt; read the server's own
decode speed (timings.predicted_per_second) and draft statistics. Output tokens are compared with llama.cpp's own
non-speculative run ("none"), its lossless check.
Usage: python b6_llamacpp.py rocm|vulkan [config ...]
"""
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b6_prompts  # noqa: E402

TAG = os.environ.get("B6_TAG", "b12380")   # official build tag for rocm/vulkan backends
NEW, PORT = 96, 8091
REPS = int(os.environ.get("B6_REPS", "3"))           # CAP1: fewer repetitions for slow (offloaded) targets
NGL = os.environ.get("B6_NGL", "99")                  # CAP1: layers on the GPU (partial offload); 99 = all
MIN_TOKS = float(os.environ.get("B6_MIN_TOKS", "50"))  # CAP1: GPU-use sanity threshold (lower for offloaded targets)
TARGET = HERE / "models" / os.environ.get("B6_TARGET", "qwen3-1.7b-bf16.gguf")
DRAFT_Q8 = HERE / "models" / "qwen3-0.6b-q8_0.gguf"
DRAFT_BF16 = HERE / "models" / "qwen3-0.6b-bf16.gguf"


def draft(n, model=DRAFT_Q8, extra=()):
    return ["--spec-type", "draft-simple", "-md", str(model), "--spec-draft-ngl", "99", "--spec-draft-n-max", str(n),
            *extra]


CONFIGS = {
    "none": [],
    "draft3": draft(3),
    "draft5": draft(5),
    "draft7": draft(7),
    "draft7_pmin": draft(7, extra=("--spec-draft-p-min", "0.75")),
    "draft5_bf16": draft(5, DRAFT_BF16),
    "ngram_mod": ["--spec-type", "ngram-mod"],
    "draft5_ngram": ["--spec-type", "draft-simple,ngram-mod", "-md", str(DRAFT_Q8), "--spec-draft-ngl", "99",
                     "--spec-draft-n-max", "5"],
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def post(path, body, timeout=300):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def healthy():
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200
    except Exception:
        return False


def start_server(exe, name, tag):
    args = [str(exe), "-m", str(TARGET), "-ngl", NGL, "-fa", os.environ.get("B6_FA", "on"), "-c", "4096", "-np", "1",
            "--host", "127.0.0.1", "--port", str(PORT), *CONFIGS[name]]
    slog = open(HERE / "results" / f"B6_server_{tag}_{name}.log", "a", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(args, stdout=slog, stderr=subprocess.STDOUT, cwd=str(exe.parent))
    t0 = time.time()
    while not healthy():
        if proc.poll() is not None or time.time() - t0 > 240:
            stop_server(proc, slog)
            return None, None
        time.sleep(0.5)
    return proc, slog


def stop_server(proc, slog):
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except Exception:
        proc.kill()
    slog.close()
    time.sleep(1)


def cold(exe, backend, names, prompts, tok):
    """B6c: every measured request on a FRESHLY STARTED server, so nothing is remembered from earlier requests.
    Each start is warmed up with one short unrelated request (GPU kernels, graphs), then the prompt is measured once."""
    tag = f"{backend}_cold{os.environ.get('B6_SUFFIX', '')}"
    warm_ids = tok("Hello! In one sentence, what is the boiling point of water?", add_special_tokens=False)["input_ids"]
    body = {"n_predict": NEW, "temperature": 0, "top_k": 1, "ignore_eos": True, "cache_prompt": False,
            "return_tokens": True, "seed": 0}
    out = {"backend": backend, "mode": "cold", "rows": [], "target": TARGET.name}
    ref_tokens = {}
    hybrid = os.environ.get("B6_COLD_ONLY_NGRAM") == "1"   # S4B3: only n-gram configs remember across requests
    for name in names:
        shared = hybrid and "ngram" not in name
        sproc, sslog = (start_server(exe, name, tag) if shared else (None, None))
        if shared and sproc is None:
            log(f"B6c {backend} {name}: server failed to start")
            continue
        for label, ids in prompts:
            speeds, dn, da, toks = [], [], [], None
            if shared:                                   # one server, same-prompt warm-up + 3 measured (B6b method)
                post("/completion", {**body, "prompt": ids})
                for _ in range(REPS):
                    r = post("/completion", {**body, "prompt": ids})
                    tm = r.get("timings", {})
                    speeds.append(tm.get("predicted_per_second", 0.0))
                    dn.append(tm.get("draft_n", 0)); da.append(tm.get("draft_n_accepted", 0))
                    toks = r.get("tokens")
            for _ in range(0 if shared else REPS):
                proc, slog = start_server(exe, name, tag)
                if proc is None:
                    log(f"B6c {backend} {name}: server failed to start")
                    break
                try:
                    post("/completion", {**body, "prompt": warm_ids, "n_predict": 16})
                    r = post("/completion", {**body, "prompt": ids})
                    tm = r.get("timings", {})
                    speeds.append(tm.get("predicted_per_second", 0.0))
                    dn.append(tm.get("draft_n", 0)); da.append(tm.get("draft_n_accepted", 0))
                    toks = r.get("tokens")
                finally:
                    stop_server(proc, slog)
            if not speeds:
                continue
            sp = statistics.median(speeds)
            if name == "none":
                ref_tokens[label] = toks
            ref = ref_tokens.get(label)
            diff = None if (ref is None or toks is None) else next(
                (i for i, (a, b) in enumerate(zip(toks, ref)) if a != b), None)
            out["rows"].append({"config": name, "prompt": label, "tok_s": sp, "tok_s_all": speeds, "draft_n": dn,
                                "draft_accepted": da, "tokens": toks, "first_diff_vs_none": diff})
            log(f"B6c {'1SRV' if shared else 'COLD'} llama.cpp {name:13s} {label:<26} {sp:6.1f} tok/s (runs {[round(x) for x in speeds]})"
                f" | draft accepted {da}/{dn} | first diff vs none: {diff}")
            if name == "none" and sp < 20:
                log(f"RESULT B6c: ABORTED - {sp:.1f} tok/s is CPU-level")
                if shared:
                    stop_server(sproc, sslog)
                return 2, out
        if shared:
            stop_server(sproc, sslog)
        rs = [r for r in out["rows"] if r["config"] == name]
        if rs:
            log(f"B6c MEAN {'ONE-SERVER' if shared else 'COLD'} llama.cpp {name}: {statistics.mean(r['tok_s'] for r in rs):.1f} tok/s mean, "
                f"{statistics.median(r['tok_s'] for r in rs):.1f} median over {len(rs)} prompts | identical to none: "
                f"{sum(r['first_diff_vs_none'] is None for r in rs)}/{len(rs)}")
    return 0, out


def main():
    backend = sys.argv[1]
    names = sys.argv[2:] or list(CONFIGS)
    if backend == "lemonade":
        ltag = (HERE / "tools" / "lemonade_latest.txt").read_text().strip()
        exe = next((HERE / "tools" / f"llama-lemonade-{ltag}").rglob("llama-server.exe"), None)
    else:
        exe = next((HERE / "tools" / f"llama-{backend}-{TAG}").rglob("llama-server.exe"), None)
    if exe is None:
        log(f"RESULT B6 {backend}: llama-server.exe not found - run get_llamacpp_latest.py first")
        return 1
    hp = subprocess.run([str(exe), "--help"], capture_output=True, text=True, errors="replace", cwd=str(exe.parent))
    spec_lines = [l.strip() for l in (hp.stdout + hp.stderr).splitlines() if "spec" in l.lower() or "draft" in l.lower()]
    log(f"B6 {backend}: {exe} | speculative options in --help: " + " || ".join(spec_lines[:40]))
    ver = subprocess.run([str(exe), "--version"], capture_output=True, text=True, errors="replace", cwd=str(exe.parent))
    log(f"B6 {backend}: version: {(ver.stdout + ver.stderr).strip()[:300]}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    prompts = b6_prompts.prompts(tok)
    if os.environ.get("B6_COLD") == "1":
        rc, out = cold(exe, backend, names, prompts, tok)
        out["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        (HERE / "results" / f"B6c_llamacpp_{backend}_cold{os.environ.get('B6_SUFFIX', '')}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        return rc
    out = {"backend": backend, "tag": TAG, "rows": [], "target": TARGET.name}
    ref_tokens = {}
    for name in names:
        args = [str(exe), "-m", str(TARGET), "-ngl", NGL, "-fa", os.environ.get("B6_FA", "on"), "-c", "4096", "-np", "1",
                "--host", "127.0.0.1", "--port", str(PORT), *CONFIGS[name]]
        slog = open(HERE / "results" / f"B6_server_{backend}{os.environ.get('B6_SUFFIX', '')}_{name}.log", "w", encoding="utf-8", errors="replace")
        proc = subprocess.Popen(args, stdout=slog, stderr=subprocess.STDOUT, cwd=str(exe.parent))
        try:
            t0 = time.time()
            while not healthy():
                if proc.poll() is not None or time.time() - t0 > 240:
                    log(f"B6 {backend} {name}: server failed to start (exit {proc.poll()}) - see "
                        f"results/B6_server_{backend}_{name}.log")
                    break
                time.sleep(1)
            else:
                slog.flush()
                dev_lines = [l.strip()[:160] for l in open(slog.name, encoding="utf-8", errors="replace")
                             if any(w in l for w in ("ROCm", "HIP", "Vulkan", "offload", "device", "gfx"))]
                log(f"B6 {backend} {name}: server device lines: " + " || ".join(dev_lines[:8]))
                body = {"n_predict": NEW, "temperature": 0, "top_k": 1, "ignore_eos": True, "cache_prompt": False,
                        "return_tokens": True, "seed": 0}
                for label, ids in prompts:
                    post("/completion", {**body, "prompt": ids})               # warm-up
                    speeds, toks, dn, da = [], None, [], []
                    for _ in range(REPS):
                        r = post("/completion", {**body, "prompt": ids})
                        tm = r.get("timings", {})
                        speeds.append(tm.get("predicted_per_second", 0.0))
                        dn.append(tm.get("draft_n", 0)); da.append(tm.get("draft_n_accepted", 0))
                        toks = r.get("tokens")
                    sp = statistics.median(speeds)
                    if name == "none":
                        ref_tokens[label] = toks
                    ref = ref_tokens.get(label)
                    diff = None if (ref is None or toks is None) else next(
                        (i for i, (a, b) in enumerate(zip(toks, ref)) if a != b), None)
                    acc = (statistics.median(da) / statistics.median(dn)) if statistics.median(dn) else None
                    out["rows"].append({"config": name, "prompt": label, "tok_s": sp, "tok_s_all": speeds,
                                        "draft_n": dn, "draft_accepted": da, "tokens": toks, "first_diff_vs_none": diff})
                    log(f"B6 llama.cpp {backend:6s} {name:13s} {label:<26} {sp:6.1f} tok/s" +
                        (f" | draft accepted {statistics.median(da):.0f}/{statistics.median(dn):.0f}" if acc else "") +
                        f" | first diff vs none: {diff}")
                rs = [r for r in out["rows"] if r["config"] == name]
                if name == "none" and statistics.mean(r["tok_s"] for r in rs) < MIN_TOKS:
                    log(f"RESULT B6 {backend}: ABORTED - non-speculative speed "
                        f"{statistics.mean(r['tok_s'] for r in rs):.1f} tok/s is CPU-level; the GPU is not being used")
                    (HERE / "results" / f"B6_llamacpp_{backend}{os.environ.get('B6_SUFFIX', '')}.json").write_text(
                        json.dumps(out, indent=1), encoding="utf-8")
                    return 2
                log(f"B6 MEAN llama.cpp {backend} {name}: {statistics.mean(r['tok_s'] for r in rs):.1f} tok/s over "
                    f"{len(rs)} prompts | identical to none: {sum(r['first_diff_vs_none'] is None for r in rs)}/{len(rs)}")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()
            slog.close()
            time.sleep(2)
    means = {n: statistics.mean(r["tok_s"] for r in out["rows"] if r["config"] == n)
             for n in names if any(r["config"] == n for r in out["rows"])}
    if means:
        best = max(means, key=means.get)
        log(f"RESULT B6 llama.cpp {backend}: best config {best} {means[best]:.1f} tok/s mean | none "
            f"{means.get('none', float('nan')):.1f} tok/s")
    out["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    (HERE / "results" / f"B6_llamacpp_{backend}{os.environ.get('B6_SUFFIX', '')}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
