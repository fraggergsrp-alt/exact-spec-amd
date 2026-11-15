"""CAP1 step 1: feasibility of running Qwen3-8B EXACTLY (lossless 16-bit) on a 16 GB card (pre-reg: docs/PREREG_CAP1.md).
Read-only analysis of the downloaded checkpoint (Qwen/Qwen3-8B safetensors, bf16), one tensor at a time:
  1. Is the bf16 -> fp16 conversion our engine uses exact for every weight? (count weights that change)
  2. F12 size (our lossless 12-bit format, kernel_f12.py): 12 bits/weight + exceptions (weights outside the per-matrix
     16-exponent window or with nonzero low mantissa bits after conversion) stored as (column, fp16 bits) + row offsets.
  3. A VRAM budget for: F12 target layers + lm_head (+ embeddings on GPU or in system RAM) + KV cache + 0.6B int4 draft.
Also records total system RAM and disk. Writes results/CAP1_check.json.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ps(cmd):
    try:
        return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as ex:
        return f"error {ex}"


def main():
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    path = Path(snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]))
    cfg = json.loads((path / "config.json").read_text())
    ram_gb = ps("[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)")
    log(f"CAP1 system RAM {ram_gb} GB | model path {path}")
    log(f"CAP1 Qwen3-8B config: layers {cfg['num_hidden_layers']}, hidden {cfg['hidden_size']}, mlp {cfg['intermediate_size']}, "
        f"heads {cfg['num_attention_heads']}/{cfg['num_key_value_heads']}, vocab {cfg['vocab_size']}, tied embeddings "
        f"{cfg.get('tie_word_embeddings')}, dtype {cfg.get('torch_dtype')}")
    dev = torch.device("cuda")
    tot = {"params": 0, "bf16_bytes": 0, "f12_bytes": 0, "not_fp16_exact": 0, "exceptions": 0, "f12_layers": 0,
           "f12_lm_head": 0, "bf16_embed": 0, "small_bytes": 0}
    worst_exc_row, worst_name = 0, ""
    per = []
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as sf:
            for name in sf.keys():
                t = sf.get_tensor(name)
                n = t.numel()
                tot["params"] += n
                tot["bf16_bytes"] += n * 2
                if t.dim() != 2 or "norm" in name:
                    tot["small_bytes"] += n * 2
                    continue
                g = t.to(dev)
                h = g.to(torch.float16)
                not_exact = int((h.to(torch.bfloat16).view(torch.int16) != g.view(torch.int16)).sum())
                tot["not_fp16_exact"] += not_exact
                if "embed_tokens" in name:
                    tot["bf16_embed"] += n * 2
                    del g, h
                    continue
                u = h.view(torch.int16).to(torch.int32) & 0xFFFF
                E, m = (u >> 10) & 31, u & 1023
                ok = (m & 7) == 0
                hist = torch.bincount(E[ok].reshape(-1).long(), minlength=32)
                cum = torch.cat([torch.zeros(1, dtype=torch.int64, device=dev), torch.cumsum(hist, 0)])
                e0 = int(torch.argmax(torch.stack([cum[e + 16] - cum[e] for e in range(17)])))
                fit = (E >= e0) & (E <= e0 + 15) & ok
                exc_row = (~fit).sum(1)
                n_exc = int(exc_row.sum())
                N = h.shape[0]
                size = n * 3 // 2 + n_exc * 4 + (N + 1) * 4
                tot["f12_bytes"] += size
                tot["exceptions"] += n_exc
                if "lm_head" in name:
                    tot["f12_lm_head"] += size
                else:
                    tot["f12_layers"] += size
                if int(exc_row.max()) > worst_exc_row:
                    worst_exc_row, worst_name = int(exc_row.max()), name
                per.append({"name": name, "shape": list(h.shape), "e0": e0, "exceptions": n_exc,
                            "max_exc_row": int(exc_row.max()), "not_fp16_exact": not_exact})
                del g, h, u, E, m, fit
        torch.cuda.empty_cache()
    gb = 1e9
    L, H, kvh, hd = cfg["num_hidden_layers"], cfg["hidden_size"], cfg["num_key_value_heads"], cfg.get("head_dim", 128)
    kv_per_tok = L * 2 * kvh * hd * 2
    budget = {"f12_layers_gb": tot["f12_layers"] / gb, "f12_lm_head_gb": tot["f12_lm_head"] / gb,
              "embed_bf16_gb": tot["bf16_embed"] / gb, "kv_cache_2k_tokens_gb": kv_per_tok * 2048 / gb,
              "draft_0p6b_int4_estimate_gb": 0.75, "activations_graphs_estimate_gb": 0.8}
    budget["total_embed_on_gpu_gb"] = sum(v for k, v in budget.items() if k.endswith("_gb"))
    budget["total_embed_in_ram_gb"] = budget["total_embed_on_gpu_gb"] - budget["embed_bf16_gb"]
    out = {"ram_gb": ram_gb, "config": {k: cfg.get(k) for k in ("num_hidden_layers", "hidden_size", "intermediate_size",
           "num_attention_heads", "num_key_value_heads", "vocab_size", "tie_word_embeddings", "torch_dtype")},
           "totals": tot, "worst_exceptions_per_row": worst_exc_row, "worst_matrix": worst_name,
           "budget_A_modeled": budget, "matrices": per}
    (ROOT / "results" / "CAP1_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"CAP1 params {tot['params'] / 1e9:.3f} B | bf16 size {tot['bf16_bytes'] / gb:.2f} GB | weights changed by bf16->fp16: "
        f"{tot['not_fp16_exact']} ({100 * tot['not_fp16_exact'] / tot['params']:.6f}%)")
    log(f"CAP1 F12: layers {budget['f12_layers_gb']:.2f} GB + lm_head {budget['f12_lm_head_gb']:.2f} GB | exceptions "
        f"{tot['exceptions']} ({100 * tot['exceptions'] / max(1, tot['params']):.4f}%), worst row {worst_exc_row} ({worst_name})")
    log(f"CAP1 VRAM budget (A, modeled): embeddings on GPU {budget['total_embed_on_gpu_gb']:.2f} GB | embeddings in system RAM "
        f"{budget['total_embed_in_ram_gb']:.2f} GB | card 15.9 GB, ~1.4 GB used by Windows/desktop (B, PERF1)")
    verdict = "FITS" if budget["total_embed_in_ram_gb"] < 15.9 - 1.4 - 0.5 else "DOES NOT FIT"
    log(f"RESULT CAP1-check: F12 Qwen3-8B {verdict} (with embeddings in system RAM); bf16->fp16 exact: "
        f"{tot['not_fp16_exact'] == 0}")


if __name__ == "__main__":
    main()
