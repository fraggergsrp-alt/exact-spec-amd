"""Generate text with the exact engine.

Examples:
  python generate.py "Explain what a hash table is."
  python generate.py "Write a short poem about the sea." --temp 0.7 --seed 1
  python generate.py --chat                      (interactive, multi-turn)
  python generate.py "Count to 20." --compare    (runs speculative AND plain decoding, checks they are identical)

Options: --model (default Qwen/Qwen3-8B), --max-new N, --temp T (0 = greedy), --seed S, --no-spec, --raw (no chat
template). Temperature and seed are fixed for the whole session.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "exactspec"))
from exact8b import EOS_IDS, ExactEngine


def main():
    ap = argparse.ArgumentParser(description="Exact (lossless, speculation-invariant) Qwen3 generation")
    ap.add_argument("prompt", nargs="?", help="prompt text (omit with --chat)")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-spec", action="store_true", help="plain decoding (no draft model)")
    ap.add_argument("--raw", action="store_true", help="do not apply the chat template")
    ap.add_argument("--chat", action="store_true", help="interactive multi-turn chat")
    ap.add_argument("--compare", action="store_true", help="run speculative and plain decoding and compare the tokens")
    a = ap.parse_args()
    if not a.chat and not a.prompt:
        ap.error("give a prompt, or use --chat")

    eng = ExactEngine(a.model, temperature=a.temp, seed=a.seed)

    if a.chat:
        msgs = []
        print("Type a message (empty line or Ctrl+C to quit).")
        while True:
            try:
                user = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user:
                break
            msgs.append({"role": "user", "content": user})
            ids = eng.encode_chat(msgs)
            print("model> ", end="", flush=True)
            shown = [0]
            buf = []

            def on_tokens(new):
                buf.extend(new)
                text = eng.tok.decode([t for t in buf if t not in EOS_IDS], skip_special_tokens=True)
                print(text[shown[0]:], end="", flush=True)
                shown[0] = len(text)
            out, st = eng.generate_ids(ids, a.max_new, speculative=not a.no_spec, on_tokens=on_tokens)
            reply = eng.tok.decode([t for t in out if t not in EOS_IDS], skip_special_tokens=True)
            msgs.append({"role": "assistant", "content": reply})
            print(f"\n[{st['tokens']} tokens, {st['tok_s']:.1f} tok/s]")
        return

    ids = eng.tok(a.prompt)["input_ids"] if a.raw else eng.encode_chat([{"role": "user", "content": a.prompt}])
    if a.compare:
        spec, s1 = eng.generate_ids(ids, a.max_new, speculative=True)
        plain, s2 = eng.generate_ids(ids, a.max_new, speculative=False)
        same = spec == plain
        print(eng.tok.decode([t for t in spec if t not in EOS_IDS], skip_special_tokens=True))
        print(f"\nspeculative {s1['tok_s']:.1f} tok/s | plain {s2['tok_s']:.1f} tok/s | {len(spec)} vs {len(plain)} tokens | "
              f"identical: {same}")
        sys.exit(0 if same else 1)
    text = eng.generate(a.prompt, a.max_new, chat=not a.raw, speculative=not a.no_spec, stream=True)
    st = eng.last_stats
    print(f"\n[{st['tokens']} tokens, {st['tok_s']:.1f} tok/s, {'speculative' if st['speculative'] else 'plain'}]")


if __name__ == "__main__":
    main()
