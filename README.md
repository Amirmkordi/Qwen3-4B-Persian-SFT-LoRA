# LoRA Fine-Tuned Qwen3-4B for Persian Instruction-Following

A parameter-efficient fine-tuning (PEFT) project that adapts **Qwen3-4B** to a Persian instruction-style task: given a category and a starting letter, the model responds with a single Persian word that matches both constraints.

The goal of this project is to do a LLM fine-tuning pipeline with synthetic dataset generation, LoRA training on Apple Silicon (MPS), adapter merging, and quantitative evaluation against the base model entirely in Persian.

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Training Configuration](#training-configuration)
- [Evaluation Methodology](#evaluation-methodology)
- [Files](#files)

---

## Overview

The task is a Persian word game:

> **Prompt:** «یک {دسته} که با حرف {حرف} شروع شود بگو»
> *(“Name a {category} that starts with the letter {letter}.”)*
>
> **Response:** a single Persian word belonging to the requested category and beginning
> with the requested letter.

For example:

| Prompt | Response |
|---|---|
| یک میوه که با حرف س شروع شود بگو | سیب (apple) |
| یک مکان که با حرف ش شروع شود بگو | شهرداری (municipality) |
| یک عضو بدن که با حرف س شروع شود بگو | ساق (shin) |

The dataset spans **44 categories** (animals, fruits, vehicles, body parts, emotions, professions, astronomy, etc.) and **33 Persian letters**, yielding **1,538 unique instruction response pairs**.

Qwen3-4B is fine-tuned with **LoRA** (rank 16) on the attention and MLP projections, on an Apple Silicon Mac using the MPS backend. Only ~0.6% of the model's parameters are trained; the rest stay frozen. After training, the LoRA adapter is merged back into the base weights to produce a standalone, deployable model.

---

## Results

Evaluation was run on a held out **10% test split, 154 examples**, seeded for reproducibility. An answer is counted correct if conditions hold:

1. it begins with the requested letter,
2. it belongs to the requested category,
3. it belongs to the Persian language.

This is a **letter-and-category** accuracy (not exact-string match), which is the right way for this open-ended task, where there are many valid words for a given letter + category, and the model is rewarded for finding a valid one.

| Model | Correct | Accuracy |
|---|---|---|
| **Base Qwen3-4B** | 39 / 154 | **25.32%** |
| **Fine-tuned (LoRA)** | 125 / 154 | **81.16%** |

The fine-tuned model improves accuracy by **~56 percentage points** over the base model.

### Why the base model scores low

The base Qwen3-4B is a thinking model: it emits long English reasoning and often answers in English or invents transliterated words, so its first token frequently fails the Persian letter and category checks. The fine-tuned model, by contrast, learns to respond directly in Persian with a single valid word.

### Qualitative examples (fine-tuned)

| Prompt | Expected | Generated | ✓ |
|---|---|---|---|
| یک وسیله الکترونیکی که با حرف م شروع شود بگو | میکروفون | مودم | ✓ |
| یک نوشیدنی که با حرف ق شروع شود بگو | قهوه | قهوه | ✓ |
| یک غذا که با حرف پ شروع شود بگو | پلو | پلو | ✓ |

Full per-example transcripts are in [`Accuracy.txt`](Accuracy.txt) (per-example T/F) and
[`eval_results.txt`](eval_results.txt) (raw generations for both models).

---

## How It Works


1. **`generate_data.py`** builds the synthetic dataset from Persian word lists, pairing every (category, word) with the word's first letter to form an instruction.
2. **`train.py`** fine-tunes Qwen3-4B with LoRA, formatting each example in the model's native Qwen chat template (`<|im_start|>…<|im_end|>`).
3. **`merge.py`** merges the trained LoRA adapter back into the base model weights, producing a standalone merged model with no adapter overhead.
4. **`eval.py`** runs both the base and merged models on the held-out test split and writes generations to `eval_results.txt`.

---

## Project Structure

```
.
├── README.md
├── requirements.txt        # pinned dependencies (exported from venv)
├── .gitignore
├── generate_data.py        # builds data/dataset.jsonl from word lists
├── train.py                # LoRA fine-tuning on Qwen3-4B (MPS)
├── merge.py                # merges LoRA adapter into base weights
├── eval.py                 # evaluates base vs fine-tuned model
├── Accuracy.txt            # per-example T/F + accuracy summary
├── eval_results.txt        # raw generations for all 154 test examples
├── data/
│   └── dataset.jsonl       # 1,538 instruction–response pairs
├── qwen3-4b/               # base model (download from Hugging Face)
├── output/                 # LoRA adapter + checkpoints (NOT in github)
└── qwen3-4b-merged/        # merged model (regenerate with merge.py)
```

> **Note on model weights:** The base model, merged model, and LoRA adapter are **not
> committed** (each is several GB, and the adapter alone is 126 MB, which is above GitHub's file
> limit). See [Setup](#setup) for how to obtain the base model and how to regenerate the adapter and merged model.

---

## Requirements

- **Python 3.11**
- **Apple Silicon Mac** (this project uses the MPS backend; CUDA is not used)
- ~20 GB free disk for the base model + checkpoints
- See [`requirements.txt`](requirements.txt) for the full pinned set. Key versions:

| Package | Version |
|---|---|
| torch | 2.10.0 |
| transformers | 5.1.0 |
| peft | 0.18.1 |
| datasets | 4.5.0 |
| accelerate | 1.12.0 |
| tokenizers | 0.22.2 |

---

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

#    Download the base model (Qwen3-4B) into ./qwen3-4b
#    Either via huggingface-cli:
huggingface-cli download Qwen/Qwen3-4B --local-dir ./qwen3-4b
#    …or use git-lfs:
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-4B ./qwen3-4b

#    (Optional) If any step needs Hub authentication, export a token:
export HF_TOKEN=hf_your_token_here
```
---

## Training Configuration

Defined in [`train.py`](train.py):

| Setting | Value |
|---|---|
| Base model | Qwen3-4B (`./qwen3-4b`) |
| Precision | float16 (MPS) |
| PEFT method | LoRA |
| LoRA rank (`r`) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Epochs | 3 |
| Batch size | 1 × 4 (gradient accumulation) → effective 4 |
| Learning rate | 1e-4, cosine schedule |
| Warmup steps | 50 |
| Max sequence length | 128 |
| Gradient checkpointing | enabled |
| Train/eval split | 90% / 10% (seed 85) |

**Chat template:** examples are formatted with Qwen3's native special tokens
(`<|im_start|>` / `<|im_end|>`), so the fine-tuned model stays compatible with the
standard Qwen prompting convention and learns to emit a real stop token.

---

## Evaluation Methodology

Defined in [`eval.py`](eval.py):

- The dataset is split 90/10 with `seed=85`; the **154-example** test set is held out from training.
- Each example is prompted with the same chat-template format used in training (`<|im_start|>assistant\n` as the generation prompt).
- Decoding is **greedy** (`do_sample=False`, `temperature=1.0`, `top_p=1.0`) for reproducibility.
- Both the base model and the merged fine-tuned model are evaluated on the same split.
- An answer is **correct** iff its first token starts with the requested letter **and** the word belongs to the requested category (see [Results](#results)).

The base model is loaded fresh and the fine-tuned model is the merged adapter — both run
on MPS, with the MPS cache cleared between the two runs.

---

## Files

| File | Purpose |
|---|---|
| [`generate_data.py`](generate_data.py) | Defines 44 Persian category word-lists, the 33-letter Persian alphabet, and `category_display` (readable labels). Pairs each unique word with its first letter to produce `data/dataset.jsonl`. |
| [`train.py`](train.py) | Loads Qwen3-4B, attaches a LoRA adapter, formats the dataset with the native chat template and label masking, and trains with the Hugging Face `Trainer`. Saves the adapter to `output/`. |
| [`merge.py`](merge.py) | Loads the base model + trained adapter, calls `merge_and_unload()`, and saves a merged model to `qwen3-4b-merged/`. |
| [`eval.py`](eval.py) | Evaluates the base and merged models on the 154-example test split; writes generations to `eval_results.txt`. |
| [`data/dataset.jsonl`](data/dataset.jsonl) | The 1,538 training examples, one JSON object per line (`{"text": "کاربر: …\nدستیار: …"}`). |
| [`Accuracy.txt`](Accuracy.txt) | human made T/F judgment and the accuracy summary for both models. |
| [`eval_results.txt`](eval_results.txt) | The full raw generations (question / expected / generated) for every test example, for both models. |
| [`requirements.txt`](requirements.txt) | All pinned Python dependencies, exported from the project venv. |

---

## Acknowledgements

- Base model: [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) by the Qwen team.
- Fine-tuning: [PEFT / LoRA](https://github.com/huggingface/peft) and
  [Transformers](https://github.com/huggingface/transformers) by Hugging Face.
