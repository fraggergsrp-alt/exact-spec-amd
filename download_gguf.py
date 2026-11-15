"""Phase E1 helper: download the two GGUF model files used for the llama.cpp baseline.

1. Qwen/Qwen3-0.6B-GGUF  Q8_0  (official, 8-bit, ~639 MB)
2. unsloth/Qwen3-0.6B-GGUF BF16 (16-bit, ~1.2 GB) - same byte count as our fp16 runs.
   Qwen3's original weights are BF16, so this is the closest precision match.

Files go into models\\ inside the project. Prints the exact paths it used.
"""
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

HERE = Path(__file__).resolve().parent
OUT = HERE / "models"
OUT.mkdir(exist_ok=True)

import sys
SIZE = sys.argv[1] if len(sys.argv) > 1 else "0.6B"   # B2: also "1.7B"
WANTED = [(f"Qwen/Qwen3-{SIZE}-GGUF", "Q8_0", f"qwen3-{SIZE.lower()}-q8_0.gguf"),
          (f"unsloth/Qwen3-{SIZE}-GGUF", "BF16", f"qwen3-{SIZE.lower()}-bf16.gguf")]

ONLY = sys.argv[2] if len(sys.argv) > 2 else None   # S4B: e.g. "BF16" to fetch just that file
for repo, tag, local_name in WANTED:
    if ONLY and tag != ONLY:
        continue
    files = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf") and tag.lower() in f.lower()]
    if not files:
        raise SystemExit(f"No {tag} .gguf file found in {repo}")
    chosen = sorted(files, key=len)[0]  # shortest name = the plain file, not a split/variant
    target = OUT / local_name
    if not target.exists():
        path = Path(hf_hub_download(repo, chosen, local_dir=OUT))
        path.replace(target)  # rename to a simple fixed name
    print(f"{repo} :: {chosen} -> {target} ({target.stat().st_size / 2**20:.0f} MiB)")
print("Done.")
