"""Phase D: a REAL small model generating text on the AMD GPU.

Model: Qwen/Qwen3-0.6B (Apache-2.0, ~0.6B parameters, 32k context).
Question: in real inference, how much do time and memory depend on prompt
length, and does the attention implementation ("eager" = standard full-matrix
attention vs "sdpa" = PyTorch's efficient kernel) matter end to end?

Measured per (attention implementation, prompt length):
  prefill_ms      time to process the whole prompt once (~ time to first token)
  decode_tok_s    tokens/second while generating new tokens one by one
  peak_alloc_mib  peak memory allocated by PyTorch during generation
Quality checks:
  - a fixed factual prompt must continue with "Paris"
  - greedy (deterministic) outputs of eager vs sdpa are compared token by token

Timing uses wall clock with torch.cuda.synchronize() (the GPU event timer was
shown to be unreliable on this Windows/ROCm setup in G001/G002).

Usage (GPU environment, after installing transformers):
  .venv-gpu\\Scripts\\python.exe inference_benchmark.py
"""
from __future__ import annotations
import argparse
import csv
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3-0.6B"
# Neutral filler text written for this lab, repeated to reach exact prompt lengths.
FILLER = ("The library opened early on Monday. Students carried notebooks, maps and "
          "small lamps between the long wooden tables while the river outside moved "
          "slowly past the old stone bridge. ")


def sync():
    torch.cuda.synchronize()


def make_prompt_ids(tok, n_tokens: int, device):
    ids = []
    while len(ids) < n_tokens:
        ids += tok(FILLER, add_special_tokens=False)["input_ids"]
    return torch.tensor([ids[:n_tokens]], device=device)


def timed(fn, warmup: int, repeat: int) -> list[float]:
    for _ in range(warmup):
        fn()
    sync()
    out = []
    for _ in range(repeat):
        sync()
        t = time.perf_counter()
        fn()
        sync()
        out.append((time.perf_counter() - t) * 1000)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-model inference benchmark (ROCm PyTorch)")
    parser.add_argument("--lengths", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    parser.add_argument("--impls", nargs="+", default=["eager", "sdpa"], choices=["eager", "sdpa"])
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1, help="identical prompts processed together")
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face model id (default Qwen/Qwen3-0.6B)")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()
    if max(args.lengths) > 16384 or min(args.lengths) < 16:
        parser.error("lengths must be between 16 and 16384")
    if not torch.cuda.is_available():
        sys.exit("No GPU visible to PyTorch. Run gpu_diagnostic.py first.")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.model)
    records, greedy_outputs, quality = [], {}, {}

    for impl in args.impls:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, attn_implementation=impl).to(device=device, dtype=torch.float16).eval()
        weights_mib = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20

        # Quality check 1: a simple factual continuation.
        fact = tok("The capital of France is", return_tensors="pt").to(device)
        with torch.no_grad():
            ans_ids = model.generate(**fact, max_new_tokens=5, do_sample=False)
        answer = tok.decode(ans_ids[0, fact["input_ids"].shape[1]:], skip_special_tokens=True)
        quality[impl] = {"answer": answer, "contains_paris": "paris" in answer.lower()}
        print(f"[{impl}] weights {weights_mib:.0f} MiB | 'The capital of France is' -> {answer!r}")

        for n in args.lengths:
            ids = make_prompt_ids(tok, n, device).repeat(args.batch, 1)
            mask = torch.ones_like(ids)

            def prefill():
                with torch.no_grad():
                    model(input_ids=ids, attention_mask=mask, use_cache=True)

            def generate():
                with torch.no_grad():
                    return model.generate(input_ids=ids, attention_mask=mask,
                                          max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens,
                                          do_sample=False, pad_token_id=tok.eos_token_id)

            try:
                prefill_ms = timed(prefill, args.warmup, args.repeat)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                gen_ms = timed(generate, 1, max(2, args.repeat // 2))
                peak_mib = (torch.cuda.max_memory_allocated() - base) / 2**20
                out = generate()
                greedy_outputs[(impl, n)] = out[0, n:].tolist()
            except torch.cuda.OutOfMemoryError:
                print(f"[{impl}] N={n}: out of GPU memory, skipped")
                torch.cuda.empty_cache()
                continue

            p, g = statistics.median(prefill_ms), statistics.median(gen_ms)
            decode_tok_s = (args.new_tokens - 1) / max((g - p) / 1000, 1e-9)  # per sequence
            step_ms = (g - p) / max(args.new_tokens - 1, 1)
            rec = {
                "model": args.model, "attn_impl": impl, "prompt_tokens": n, "new_tokens": args.new_tokens,
                "batch": args.batch,
                "decode_ms_per_step": round(step_ms, 2),
                "decode_tok_s_total": round(decode_tok_s * args.batch, 1),
                "prefill_ms_median": round(p, 2),
                "prefill_ms_min": round(min(prefill_ms), 2), "prefill_ms_max": round(max(prefill_ms), 2),
                "generate_ms_median": round(g, 2),
                "decode_tok_s_est": round(decode_tok_s, 1),
                "peak_alloc_mib_above_weights": round(peak_mib, 1),
                "weights_mib": round(weights_mib, 1),
            }
            records.append(rec)
            print(f"[{impl:>5}] N={n:>5} | prefill {p:8.2f} ms (min {min(prefill_ms):.2f}, max {max(prefill_ms):.2f}) "
                  f"| decode {step_ms:6.2f} ms/step, ~{decode_tok_s:6.1f} tok/s per seq, "
                  f"{decode_tok_s * args.batch:7.1f} total (batch {args.batch}) | peak +{peak_mib:8.1f} MiB above weights")
        del model
        torch.cuda.empty_cache()

    # Quality check 2: do eager and sdpa produce the same greedy tokens?
    agreement = {}
    if "eager" in args.impls and "sdpa" in args.impls:
        for n in args.lengths:
            a, b = greedy_outputs.get(("eager", n)), greedy_outputs.get(("sdpa", n))
            if a and b:
                same = sum(x == y for x, y in zip(a, b))
                first_diff = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
                agreement[n] = {"matching_tokens": same, "of": len(a), "first_difference_at": first_diff}
                print(f"N={n:>5}: eager vs sdpa greedy tokens identical {same}/{len(a)} "
                      f"(first difference at {first_diff})")

    outdir = HERE / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    output = {
        "project": "MEMORYWALL Phase D real-model inference",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "dtype": "float16", "batch_size": args.batch,
        "environment": {"python": sys.version.split()[0], "os": platform.platform(),
                        "torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                        "transformers": transformers.__version__,
                        "device": torch.cuda.get_device_name(0)},
        "parameters": vars(args), "quality": quality,
        "eager_vs_sdpa_agreement": {str(k): v for k, v in agreement.items()},
        "warning": "decode_tok_s is estimated as (generate - prefill) time; one machine; synthetic filler prompts.",
        "records": records,
    }
    (outdir / "inference_latest.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    if records:
        with (outdir / "inference_latest.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)
    print("Saved: results/inference_latest.json and results/inference_latest.csv")


if __name__ == "__main__":
    main()
