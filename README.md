# exact-spec-amd

**Exact speculative decoding and lossless 12-bit weights for Qwen3 on a consumer AMD GPU (RX 9070 XT, 16 GB, Windows + ROCm PyTorch).**

- Runs **Qwen3-8B with weights bit-identical to the original bf16 checkpoint**, fully inside 16 GB of VRAM, using a lossless 12-bit weight format (B12).
- **Speculative decoding output is bit-for-bit identical to plain decoding** with the same engine, for greedy decoding and for seeded sampling (temperature).
- **68–73 tok/s** for exact Qwen3-8B with speculation. That is 2.5x llama.cpp running the same full-precision weights, and **slower than llama.cpp with lossy 8-bit (Q8_0) weights**.

![Qwen3-8B decode speed: exact engine vs llama.cpp](docs/assets/qwen3-8b-speed.png)

## Quickstart

```
git clone https://github.com/fraggergsrp-alt/exact-spec-amd
cd exact-spec-amd
# 1. install a ROCm build of PyTorch for your GPU (see Requirements below), then:
pip install -r requirements.txt
# 2. generate (first run downloads Qwen3-8B, ~16 GB, and packs it to 12-bit in ~30 s):
python generate.py "Explain what a hash table is."
python generate.py "Write a short poem about the sea." --temp 0.7 --seed 1
python generate.py --chat
# 3. check exactness yourself: runs speculative AND plain decoding and compares every token
python generate.py "Count from 1 to 30 in words." --compare
```

From Python:

```python
from exact8b import ExactEngine
eng = ExactEngine("Qwen/Qwen3-8B", temperature=0.0, seed=42)
print(eng.generate("Explain what a hash table is.", max_new_tokens=200))
print(eng.last_stats)   # tokens, tok/s, rounds
```

This is a research engine, not a product. Everything was measured on one PC. Please try to replicate it on your hardware (see [Requirements](#requirements-and-reproducing-the-numbers)).

## Results (measured, one PC)

**Qwen3-8B**, 11 prompts x 96 tokens, greedy decoding, decode tok/s:

| Setup | Weights | Plain | Speculative | Spec output = plain |
|---|---|---|---|---|
| llama.cpp ROCm BF16, `-ngl 99` (spills past VRAM) | lossless | 26.9 | 18.6 | 10/11 |
| llama.cpp Vulkan BF16 (spills past VRAM) | lossless | 21.5 | 12.5 | 11/11 |
| **this engine, B12** | **lossless** | **42.7** | **68.4** | **11/11** |
| llama.cpp ROCm Q8_0 (fits) | lossy, 8-bit | 63.4 | 98.8 | 7/11 |
| llama.cpp Vulkan Q8_0 (fits) | lossy, 8-bit | 68.0 | 95.0 | 7/11 |

On 50 varied prompts x 512 tokens, exact speculative decoding reached **73.0 tok/s**, with speculative output equal to plain output on **50/50** prompts.

**Exactness** (speculative output == plain output, same engine):

| Model / test | Standard PyTorch kernels | This engine | Speed cost of exactness |
|---|---|---|---|
| Qwen3-1.7B, 50 prompts x 512 tokens, 2 sessions | 35/50 | **50/50** (both sessions) | 2.8–4.3% |
| Qwen3-4B, 50 prompts x 512 tokens, 2 sessions | 50/50 | **50/50** | 3.1–4.3% |
| Qwen3-1.7B, seeded sampling T=0.7 / 1.0 | 38/50, 36/50 | **50/50, 50/50** | 1.7–4.3% |
| Qwen3-8B (B12), 50 prompts x 512 tokens | — | **50/50** | — |

Baselines: llama.cpp ROCm (Lemonade build b1376) and llama.cpp Vulkan (official build b12426), `-c 4096`, draft Qwen3-0.6B Q8_0 with `--spec-draft-n-max 7`, same token ids. All runs used one session of 11 prompts, so the agreement counts are small samples. On this PC, Vulkan crashed until two things were switched off for the benchmark processes: the RenderDoc Vulkan layer (`VK_LOADER_LAYERS_DISABLE=VK_LAYER_RENDERDOC_Capture`) and flash attention (`-fa off`, which costs less than 2% decode speed in llama-bench). Details: [docs/PREREG_CAP1.md](docs/PREREG_CAP1.md), step 5.

**In short:** this is the fastest *lossless* way we measured to run Qwen3-8B on a 16 GB card. The lossy 8-bit format with speculative decoding is about 1.4x faster still, and in our runs llama.cpp's speculative output differed from its own plain output on 4 of 11 prompts.

## How it works

1. **Speculative decoding engine** (batch 1). A Qwen3-0.6B draft with int4 weights proposes 7 tokens, and the target checks them in one pass. It uses custom split-KV attention, fused norms/RoPE/SwiGLU, a multi-row KV write, and draft/verify steps replayed as GPU graphs.
2. **Batch-invariant matrix kernels.** Verifying 8 tokens normally rounds differently from decoding 1 token, so outputs drift. One kernel serves 1–32 rows and sums every output in a fixed per-row order: lane-strided chunks, fp32, no FMA contraction, a fixed shuffle reduction. That makes verify bit-identical to plain decoding (`kernel_bi.py`).
3. **Exact seeded sampling.** The token is the argmax of `logit/T + Gumbel(seed, position, token)`, with the noise computed from a hash rather than an RNG state. This gives an exact softmax sample that is a fixed function of each position's logits (`sampler.py`).
4. **B12, a lossless 12-bit format for bf16 weights.** Per weight: 8 bits (sign + 7 mantissa bits) plus a 4-bit exponent code within a per-row window of 15 exponents. Weights outside the window are kept in a per-row exception list with their exact bits (0.02% for Qwen3-8B). The format averages 12.0 bits/weight and decodes back to the original bits, and the kernel is batch-invariant (`research/cap1/kernel_b12.py`).
5. **Qwen3-8B on 16 GB** (`research/cap1/sd8b.py`). The bf16 model is loaded in system RAM, each matrix is packed to B12 on the GPU (about 29 s), and the embedding table stays in system RAM.

## Requirements and reproducing the numbers

Requirements:

- An AMD RDNA4 GPU with 16 GB (tested: RX 9070 XT, gfx1201)
- ROCm-enabled PyTorch (tested: 2.14 + ROCm 10.1 on Windows 11)
- Python 3.12 and about 35 GB of disk for Qwen3-8B
- At least 32 GB of system RAM for the 8B run

```
python -m venv .venv-gpu
.venv-gpu\Scripts\activate          (Linux: source .venv-gpu/bin/activate)
# install a ROCm build of PyTorch for your GPU first (see pytorch.org / AMD docs), then:
pip install -r requirements.txt
```

Run the replication scripts:

| Script | What it checks | Time |
|---|---|---|
| `scripts\replicate_8b.bat` (Linux: `scripts/replicate_8b.sh`) | Qwen3-8B exact speculative decoding on 11 prompts; downloads the model (~16 GB) on first run | ~10 min |
| `scripts\replicate_exact_1p7b.bat` | Qwen3-1.7B: exactness and speed, standard kernels vs batch-invariant kernels | ~10 min |

Each prints one `RESULT` line with plain tok/s, speculative tok/s and `spec==plain N/N`. Please share your numbers, GPU and OS in an issue.

The llama.cpp comparison uses `b6_llamacpp_cap1.py`, with llama.cpp binaries from `get_lemonade_latest.py` and models from `download_gguf.py`.

## Limitations

- One GPU, one OS (Windows 11), Qwen3 0.6B–8B only, batch size 1.
- Activations are fp16. Exactness means speculative == plain decoding of this engine with the same seed, not equality with other engines.
- In the 1.7B/4B engines, weights are converted bf16 → fp16, which changes 0.04% of weights. Only the 8B B12 path keeps matmul weights bit-identical to the checkpoint.
- Sampling supports temperature only (no top-k/top-p). Prompts up to ~1,000 tokens.
- Not independently replicated yet. Lossy 8-bit + speculation is faster.

## Prior work (none of the core ideas is new)

- Speculative decoding: Leviathan et al. 2023; Chen et al. 2023.
- Batch-invariant kernels: Thinking Machines Lab 2025; vLLM/SGLang deterministic modes; LLM-42.
- Drafter-invariant coupling with shared randomness: Daliri, Musco, Suresh 2025.
- Lossless bf16 weight compression: DFloat11 (~11 bits/weight, Huffman-coded exponents; tighter than our 12 bits), ZipServ, ZipNN.

This repository contributes the integration on consumer AMD hardware and the measurements. All hypotheses were pre-registered, failures included: see [`docs/`](docs).

## Disclosure

Most of the code and documentation was written with substantial help from an AI assistant (Claude, Anthropic), under the author's direction. All experiments were run by the author on his own hardware.

## License

MIT, see [LICENSE](LICENSE). Qwen3 models are Apache-2.0 and are downloaded from Hugging Face, not included here.
