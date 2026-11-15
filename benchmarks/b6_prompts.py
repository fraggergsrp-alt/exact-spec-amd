"""B6: the one prompt set shared by every engine in the strongest-baseline comparison (identical token ids).
5 raw prompts (as in SD3-SD6) + 6 chat prompts (Qwen3 chat template, thinking off), as in EG1."""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "exactspec")]

RAW = ["The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
       "List of the planets in our solar system, in order from the Sun:\n1.",
       "Once upon a time, in a small village by the sea,"]
CHATS = ["Write a short email to a colleague asking to move our meeting to Thursday.",
         "Explain how a refrigerator works in simple terms.",
         "Write a Python function that checks whether a string is a palindrome, with a docstring.",
         "A shop sells pens at 3 dollars each and notebooks at 5 dollars each. Tom buys 4 pens and 2 notebooks. "
         "How much does he pay? Show your steps.",
         "Summarize the plot of Romeo and Juliet in five sentences.",
         "List five tips for staying focused while studying."]


def prompts(tok):
    """[(label, [token ids])]"""
    from inference_benchmark import FILLER
    filler = []
    while len(filler) < 1024:
        filler += tok(FILLER, add_special_tokens=False)["input_ids"]
    out = [("filler 1024", filler[:1024])]
    out += [(t[:24], tok(t, add_special_tokens=False)["input_ids"]) for t in RAW]
    for c in CHATS:
        text = tok.apply_chat_template([{"role": "user", "content": c}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        out.append(("chat: " + c[:18], tok(text, add_special_tokens=False)["input_ids"]))
    return out
