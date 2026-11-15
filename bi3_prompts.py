"""BI3: 50 varied prompts for the bit-identity study (fixed before any BI3 run; never tuned on results).
1 long filler prompt + 19 raw continuations + 30 chat prompts (Qwen3 chat template, thinking off).
Categories: code, math, facts, lists, stories, dialogue, structured output (JSON / tables), translation and other
languages, summaries, repetitive text, instructions. Labels are numbered so every row is unique."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RAW = [
    "The history of the printing press begins",
    "import numpy as np\n\ndef softmax(x):\n",
    "class Stack:\n    \"\"\"A simple LIFO stack.\"\"\"\n    def __init__(self):\n",
    "SELECT name, COUNT(*) AS orders\nFROM customers\n",
    "The mitochondria is",
    "Monday: rest day\nTuesday: 5 km easy run\nWednesday:",
    "Q: What is 17 times 23?\nA:",
    "Dear Sir or Madam,\n\nI am writing to",
    "<html>\n<head>\n  <title>My Page</title>\n",
    "Il était une fois, dans un petit village,",
    "Der schnellste Weg, eine neue Sprache zu lernen, ist",
    "1, 1, 2, 3, 5, 8, 13,",
    "Chapter 1\n\nThe rain had not stopped for three days when",
    "Ingredients:\n- 2 cups flour\n- 1 cup sugar\n",
    "The theory of general relativity states that",
    "{\n  \"name\": \"Alice\",\n  \"age\": 30,\n",
    "To install the package, run the following command:",
    "A B C D E F G H I J K",
    "In conclusion, the results of this study suggest that",
]

CHATS = [
    "Write a short four-line poem about the ocean at night.",
    "What are the main differences between TCP and UDP? Answer in a short list.",
    "Write a JavaScript function that debounces another function.",
    "Solve for x: 3x + 7 = 25. Show each step.",
    "Translate into Spanish: 'The weather is nice today, let's go for a walk in the park.'",
    "Give me a JSON object describing a book with title, author, year and three tags.",
    "Explain recursion to a ten-year-old.",
    "Write a short dialogue between a customer and a barista.",
    "What causes the seasons on Earth?",
    "Write a bash one-liner that counts the lines in all .txt files in a folder.",
    "List the first ten prime numbers and explain how you know each is prime.",
    "Summarize the causes of World War I in one paragraph.",
    "Write a Python class for a bank account with deposit, withdraw and balance methods.",
    "Make a markdown table comparing cats and dogs on four traits.",
    "Give three arguments for and three against working from home.",
    "What is the derivative of x^3 * sin(x)? Show the steps.",
    "Write a limerick-style funny story in prose about a forgetful robot.",
    "Explain what a hash table is and why lookups are fast.",
    "Write a cover letter opening paragraph for a junior data analyst job.",
    "Describe the water cycle in five steps.",
    "Write a SQL query that finds the second highest salary in an employees table.",
    "A train travels 120 km in 1.5 hours. What is its average speed? Then convert it to m/s.",
    "Write a short product description for a reusable water bottle.",
    "Explain the difference between a virus and a bacterium.",
    "Write a C function that reverses a string in place.",
    "Give a step-by-step recipe for pancakes.",
    "What were the main achievements of the Roman Empire?",
    "Write a regular expression that matches email addresses and explain each part.",
    "Plan a three-day trip to Rome with one sentence per morning, afternoon and evening.",
    "Rewrite this sentence more formally: 'hey, can u send me that file asap? thx'",
]


def prompts(tok):
    """[(label, [token ids])]"""
    from inference_benchmark import FILLER
    filler = []
    while len(filler) < 1024:
        filler += tok(FILLER, add_special_tokens=False)["input_ids"]
    out = [("00 filler 1024", filler[:1024])]
    for i, t in enumerate(RAW, 1):
        out.append((f"{i:02d} raw: " + t[:16].replace("\n", " "), tok(t, add_special_tokens=False)["input_ids"]))
    for i, c in enumerate(CHATS, len(RAW) + 1):
        text = tok.apply_chat_template([{"role": "user", "content": c}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        out.append((f"{i:02d} chat: " + c[:16], tok(text, add_special_tokens=False)["input_ids"]))
    assert len(out) == 50
    return out
