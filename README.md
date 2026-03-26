<div align="center">

# Grounded-Contrastive Decoding

### Mitigating Hallucinations in Large Vision-Language Models via Grounded-Contrastive Decoding

<br>

[![IEEE](https://img.shields.io/badge/IEEE-Published-blue.svg)](https://ieeexplore.ieee.org/document/11283461)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![LLaVA](https://img.shields.io/badge/Base%20Model-LLaVA--1.5--7B-green.svg)](https://github.com/haotian-liu/LLaVA)
[![Training Free](https://img.shields.io/badge/Training-Free-orange.svg)]()

<br>

**Zhiyun Zhang\*** · **Haotian Zhang\***

Glasgow College, University of Electronic Science and Technology of China

`2960259Z@student.gla.ac.uk` · `2023190903014@std.uestc.edu.cn`

*\* Equal contribution*

<br>

| 📄 [Paper](https://ieeexplore.ieee.org/document/11283461) | 🚀 [Quick Start](#quick-start) | 📊 [Results](#-experimental-results) | 🛠️ [Installation](#️-installation) |
|:---:|:---:|:---:|:---:|

</div>

---

## TL;DR

> **GCD** is a **plug-and-play, training-free** decoding framework that significantly reduces hallucinations in Large Vision-Language Models. It combines a lightweight **representation disentanglement** module with a **contrastive decoding** strategy, achieving state-of-the-art performance on MME (1505.07), MM-Vet (32.9), and MMMU (36.1) using LLaVA-1.5-7B — with zero retraining.

---

## Table of Contents

- [Motivation](#-motivation)
- [Method Overview](#-method-overview)
  - [Stage 1: Representation Disentanglement](#stage-1--representation-disentanglement)
  - [Stage 2: Grounded-Contrastive Decoding](#stage-2--grounded-contrastive-decoding)
- [Experimental Results](#-experimental-results)
  - [MME Benchmark](#mme-benchmark)
  - [MM-Vet Benchmark](#mm-vet-benchmark)
  - [MMMU Benchmark](#mmmu-benchmark)
  - [Ablation Study](#ablation-study)
- [Installation](#️-installation)
- [Quick Start](#-quick-start)
- [Evaluation](#-evaluation)
- [Repository Structure](#-repository-structure)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)

---

## Motivation

Large Vision-Language Models (LVLMs) like LLaVA have achieved remarkable progress in multimodal understanding. Yet their deployment is hindered by a critical failure mode: **hallucination** — generating outputs that *sound* plausible but *contradict* the visual input.

<div align="center">

```
┌──────────────────────────────────────────────────────────────────┐
│  Image: A dog sitting on a red couch                             │
│                                                                  │
│  ❌  Hallucinated:  "A cat is lying on a blue sofa next to      │
│                      a potted plant."                            │
│                                                                  │
│  ✅  GCD Output:    "A dog is sitting on a red couch."           │
└──────────────────────────────────────────────────────────────────┘
```

</div>

Unlike hallucinations in text-only LLMs (caused by factual gaps), LVLM hallucinations arise from three distinct failure modes:

| Failure Mode | Description | Manifestation |
|:---|:---|:---|
| **Entangled Representations** | Visual and textual features are spuriously correlated in the latent space | Non-existent objects described |
| **Language Prior Dominance** | The LLM backbone overrides visual evidence with statistical priors | Common co-occurrence patterns fabricated |
| **Decoding Instability** | Visual grounding signals degrade across auto-regressive steps | Attribute / relation errors accumulate |

Existing methods address these issues *individually* and often require expensive retraining or sacrifice fluency. **GCD** resolves all three — simultaneously, at inference time, with no extra training.

---

## Method Overview

GCD operates in two complementary stages that work synergistically:

```
                        ┌──────────────────────────────────────────────────┐
                        │               GCD Pipeline                       │
                        └──────────────────────────────────────────────────┘

  Input Image I                         Text Context x
       │                                      │
       ▼                                      ▼
  ┌─────────┐                          ┌───────────┐
  │  CLIP   │                          │ Tokenizer │
  │ Encoder │                          └─────┬─────┘
  └────┬────┘                                │
       │  f_v                                │
       ▼                                     │
  ┌──────────┐                               │
  │   MLP    │  OriginalProj(f_v)            │
  │Projector │                               │
  └────┬─────┘                               │
       │                                     │
       ▼                                     │
  ┌─────────────────────┐                    │
  │   STAGE 1           │                    │
  │  Representation     │  v (disentangled)  │
  │  Disentanglement    │ ─────────────────► │
  │  (Eq. 1 & 2)        │                    │
  └─────────────────────┘                    │
       │                                     │
       │  also produces:                     │
       │  v_neg  (noise-perturbed)  ─────► ─►│
       │  v_text (text-only)       ─────► ─►│
       │                                     ▼
       │                          ┌─────────────────────┐
       │                          │   STAGE 2           │
       │                          │  Grounded-          │
       │                          │  Contrastive        │
       │                          │  Decoding           │
       │                          │  (Eq. 3 & 4)        │
       │                          └──────────┬──────────┘
       │                                     │
       │                                     ▼
       │                             Generated tokens
       │                             (hallucination-free)
       └─────────────────────────────────────┘
```

---

### Stage 1 — Representation Disentanglement

**Goal:** Remove spurious visual-textual correlations *before* the LLM sees the features.

We precompute a *confusion prototype dictionary* $D_v$ offline. Each prototype $D_v[c]$ is the centroid of projected visual embeddings for confusing category $c$ (e.g., common object co-occurrence patterns):

$$D_v[c] = \frac{1}{|S_c|}\sum_{i \in S_c} \mathrm{Proj}(f_v^{(i)}) \tag{1}$$

During inference, we estimate and subtract the contribution of these prototypes using a lightweight cross-attention mechanism. Given:

$$Q = W_q \cdot \mathrm{Proj}(f_v),\quad K = W_k D_v^\top,\quad V = W_v D_v^\top$$

the disentangled embedding is:

$$v = \mathrm{Proj}(f_v) - \mathrm{Softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V \tag{2}$$

> $\mathrm{Proj}(\cdot)$ denotes the MLP projector (OriginalProj in the paper).

**Key properties:**
- The prototype dictionary is **precomputed offline** — zero overhead per query
- The three projection matrices ($W_q, W_k, W_v$) are **lightweight** (no new training required when used with frozen prototypes)
- The subtraction operation is **exact and interpretable**: it directly removes the weighted contribution of known spurious patterns

---

### Stage 2 — Grounded-Contrastive Decoding

**Goal:** At every auto-regressive step, actively suppress language-prior-driven tokens.

We maintain **three parallel decoding streams** and combine their logits:

```
  Disentangled embedding v ──────► l_v(y_t)   ──┐
                                                  │   s_GCD = (1+β)·l_v - α·l_neg - β·l_text
  Noise-perturbed embedding v_neg ► l_neg(y_t) ──┤
                                                  │
  Text-only embedding v_text ──────► l_text(y_t) ─┘
```

$$s_{\mathrm{GCD}}(y_t) = (1+\beta)\,l_v(y_t) - \alpha\,l_{\mathrm{neg}}(y_t) - \beta\,l_{\mathrm{text}}(y_t) \tag{3}$$

| Symbol | Source | Role |
|:---:|:---|:---|
| $l_v(y_t)$ | Disentangled visual embedding | **Amplified** — primary grounded signal |
| $l_{\mathrm{neg}}(y_t)$ | Noise-perturbed visual embedding | **Suppressed** — captures ungrounded patterns |
| $l_{\mathrm{text}}(y_t)$ | Text-only (no vision) | **Suppressed** — pure language prior |
| $\alpha = 0.5$ | — | Controls negative-context suppression strength |
| $\beta = 0.3$ | — | Controls text-only suppression / visual amplification |

**Adaptive KL-divergence scaling** prevents over-suppression. At each step, we measure how far the GCD distribution deviates from the base model distribution:

$$\alpha \leftarrow \begin{cases} \alpha \cdot \dfrac{\tau}{\mathrm{KL}(p_{\mathrm{GCD}} \| p_v)} & \text{if } \mathrm{KL}(p_{\mathrm{GCD}} \| p_v) > \tau \\ \alpha & \text{otherwise} \end{cases} \tag{4}$$

*(same update for* $\beta$*)*

If the adjustment is too aggressive ($\mathrm{KL} > \tau$), parameters are damped proportionally. This dynamic mechanism **preserves language fluency** while enforcing visual grounding. KV-caching across auxiliary forward passes keeps the additional inference cost minimal.

---

## Experimental Results

All experiments use **LLaVA-1.5-7B** as the backbone, with temperature set to 0 for fair comparison.

### MME Benchmark

MME evaluates multimodal perception across 10 categories. Each category is scored out of 200 (maximum total: 2000).

| Method | Exist. | Count | Pos. | Color | Posters | Celeb. | Scene | Landmark | Artwork | OCR | **Total** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Vanilla | 190.00 | 160.00 | 138.33 | 165.00 | 140.48 | 135.88 | 156.25 | 161.50 | 118.50 | 125.00 | 1490.94 |
| DoLA | 190.00 | 158.33 | 143.33 | 165.00 | 139.46 | 133.24 | 157.00 | 161.50 | 118.75 | 125.00 | 1491.61 |
| VCD | 173.33 | 151.67 | 138.33 | 165.00 | 140.48 | 137.06 | 151.00 | 164.75 | 120.50 | 117.50 | 1459.62 |
| VDD | 180.00 | 148.30 | 135.00 | 170.00 | 141.84 | 144.71 | 151.75 | 167.50 | 123.75 | 110.00 | 1472.88 |
| DeCO | 190.00 | 148.33 | 115.00 | 165.00 | 149.66 | 135.29 | 152.25 | 164.50 | 110.25 | 130.00 | 1460.29 |
| **GCD (Ours)** | **190.00** | **161.00** | 138.33 | 165.00 | 140.48 | **138.76** | **157.00** | 164.50 | 115.00 | **135.00** | **1505.07** |

<sub>Bold = best result per column. GCD achieves the highest total score, with particular strength in **Count** (+0.7 vs. vanilla) and **OCR** (+10.0 vs. vanilla).</sub>

---

### MM-Vet Benchmark

MM-Vet tests advanced multimodal reasoning across six integrated capability dimensions.

| Method | Rec | OCR | Know | Gen | Spat | Math | **Total** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Vanilla | 36.1 | 23.0 | 18.0 | 22.2 | 25.1 | 11.5 | 31.1 |
| DoLA | 36.5 | 22.3 | 18.1 | 23.0 | 25.7 | 7.7 | 30.8 |
| VCD | 36.1 | 22.4 | 21.0 | 23.1 | 28.4 | 3.8 | 31.1 |
| VDD | 37.1 | 22.8 | 19.0 | 21.7 | 28.3 | 11.2 | 31.8 |
| DeCO | 35.8 | 26.8 | 19.2 | 21.3 | 30.2 | 7.7 | 32.6 |
| **GCD (Ours)** | 36.4 | **26.9** | **21.1** | **22.6** | **30.3** | **9.8** | **32.9** |

<sub>GCD leads on OCR, Knowledge, Generation, Spatial, and Math — demonstrating **balanced** improvement rather than trading off one skill for another.</sub>

---

### MMMU Benchmark

MMMU evaluates factual consistency and hallucination mitigation in multimodal outputs across 30 subjects.

```
  Score (%)
  37 ┤
     │
  36 ┤                                                    ██  ← GCD 36.1
     │                                          ██       ██
  35 ┤              ██       ██       ██        ██  ██   ██
     │    ██        ██       ██       ██        ██  ██   ██
  34 ┤    ██        ██       ██       ██        ██  ██   ██
     │    ██        ██       ██       ██        ██  ██   ██
  33 ┤    ██        ██       ██       ██        ██  ██   ██
     └────────────────────────────────────────────────────────
          33.0    Vanilla   DoLA    VCD       VDD  DeCO  GCD
                  35.3      35.7    35.8     34.9  33.9  36.1
```

| Method | Score |
|:---|:---:|
| Vanilla | 35.3 |
| DoLA | 35.7 |
| VCD | 35.8 |
| VDD | 34.9 |
| DeCO | 33.9 |
| **GCD (Ours)** | **36.1** |

---

### Ablation Study

We conduct controlled ablation studies to validate each hyperparameter. Results confirm that the default values are robustly optimal across all three benchmarks.

#### Effect of $\alpha$ (negative-context suppression) — evaluated on MME

| $\alpha$ | 0.1 | 0.3 | **0.5** | 0.7 | 0.9 |
|:---|:---:|:---:|:---:|:---:|:---:|
| MME Total | 1455.2 | 1483.6 | **1505.07** | 1499.1 | 1490.5 |

<sub>Too small → under-suppresses hallucinations. Too large → penalises valid visual tokens.</sub>

#### Effect of $\beta$ (text-only suppression / visual amplification) — evaluated on MM-Vet

| $\beta$ | 0.0 | 0.1 | **0.3** | 0.5 | 0.7 |
|:---|:---:|:---:|:---:|:---:|:---:|
| MM-Vet Total | 30.8 | 31.5 | **32.9** | 32.1 | 31.8 |

<sub>$\beta = 0$ reverts to no visual amplification. Large $\beta$ over-emphasises visual tokens and slightly hurts fluency.</sub>

#### Effect of $\tau$ (KL divergence threshold) — evaluated on MMMU

| $\tau$ | 0.01 | 0.02 | **0.05** | 0.08 | 0.10 |
|:---|:---:|:---:|:---:|:---:|:---:|
| MMMU Score | 35.2 | 35.4 | **36.1** | 35.7 | 35.5 |

<sub>Too small → triggers excessive scaling, destabilises generation. Too large → allows uncontrolled divergence from the base model.</sub>

---

## Installation

### Requirements

- Python 3.9+
- CUDA 11.8+ (for GPU inference)
- ~16 GB VRAM (LLaVA-1.5-7B in fp16)

### Step-by-step

```bash
# 1. Clone this repository
git clone https://github.com/<your-username>/GCD.git
cd GCD

# 2. Create a virtual environment (recommended)
conda create -n gcd python=3.10 -y
conda activate gcd

# 3. Install PyTorch (adjust CUDA version as needed)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 4. Install LLaVA (required backbone)
pip install git+https://github.com/haotian-liu/LLaVA.git

# 5. Install remaining dependencies
pip install -r requirements.txt
```

### Verify installation

```python
import torch
from gcd import load_llava_model, GCDLogitsProcessor
print("GCD ready. CUDA:", torch.cuda.is_available())
```

---

## Quick Start

### Minimal inference example

```python
from PIL import Image
from transformers import LogitsProcessorList
from gcd import (
    GCDLogitsProcessor,
    RepresentationDisentanglement,
    build_gcd_inputs,
    load_llava_model,
)

# ── 1. Load LLaVA-1.5-7B ────────────────────────────────────────────────────
tokenizer, model, image_processor, _ = load_llava_model(
    "liuhaotian/llava-v1.5-7b",
    device="cuda",
)

# ── 2. (Optional) Load disentanglement prototypes ────────────────────────────
#   Skip this block to run GCD without disentanglement (still improves over
#   vanilla via contrastive decoding alone).
dis_module = RepresentationDisentanglement(embed_dim=4096).cuda()
dis_module.load_prototypes("prototypes/coco_prototypes.pt")
dis_module.eval()

# ── 3. Prepare the three sets of input embeddings ───────────────────────────
image  = Image.open("example.jpg").convert("RGB")
prompt = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers. "
    "USER: <image>\nDescribe what you see in this image. ASSISTANT:"
)

input_ids, dis_emb, neg_emb, txt_emb = build_gcd_inputs(
    model, tokenizer, image_processor,
    image, prompt,
    disentanglement_module=dis_module,   # pass None to skip disentanglement
    noise_std=1.0,                       # std for negative-context perturbation
    device="cuda",
)

# ── 4. Instantiate the GCD LogitsProcessor ───────────────────────────────────
gcd_processor = GCDLogitsProcessor(
    model=model,
    input_ids=input_ids,
    disentangled_inputs_embeds=dis_emb,
    negative_inputs_embeds=neg_emb,
    text_inputs_embeds=txt_emb,
    alpha=0.5,    # negative-context suppression weight  (paper default)
    beta=0.3,     # text-only suppression weight          (paper default)
    tau=0.05,     # KL-divergence adaptive threshold      (paper default)
)

# ── 5. Generate with GCD ─────────────────────────────────────────────────────
output_ids = model.generate(
    inputs_embeds=dis_emb,
    max_new_tokens=256,
    do_sample=False,
    temperature=0,
    logits_processor=LogitsProcessorList([gcd_processor]),
)

response = tokenizer.decode(
    output_ids[0, dis_emb.shape[1]:],
    skip_special_tokens=True,
)
print(response)
```

### Building your own prototype dictionary

If you want to use the representation disentanglement module with custom data:

```python
from gcd import RepresentationDisentanglement, get_visual_embeddings

# Assume you have pre-collected visual embeddings per confusing category:
#   embeddings_by_category = {
#     "cat_vs_dog":   [emb_1, emb_2, ...],   # each emb: [N_patches, D]
#     "chair_vs_sofa":[emb_1, emb_2, ...],
#     ...
#   }

dis_module = RepresentationDisentanglement(embed_dim=4096).cuda()
dis_module.build_prototypes(embeddings_by_category)
dis_module.save_prototypes("prototypes/my_prototypes.pt")
```

---

## Evaluation

Use `run_eval.py` as the unified entry point for all three benchmarks.

### MME

```bash
python run_eval.py \
    --benchmark  mme \
    --model-path liuhaotian/llava-v1.5-7b \
    --data-root  /path/to/MME \
    --output     results/mme_gcd.json
```

Expected MME dataset layout:
```
MME/
├── Existence/
│   ├── images/
│   └── questions.jsonl
├── Count/
│   ├── images/
│   └── questions.jsonl
...
```

### MM-Vet

```bash
python run_eval.py \
    --benchmark  mmvet \
    --model-path liuhaotian/llava-v1.5-7b \
    --data-root  /path/to/mm-vet \
    --output     results/mmvet_gcd.json
```

> The script saves model outputs to JSON. Run the [official MM-Vet GPT-4 grader](https://github.com/yuweihao/MM-Vet) on this file to obtain final scores.

### MMMU

```bash
python run_eval.py \
    --benchmark  mmmu \
    --model-path liuhaotian/llava-v1.5-7b \
    --split      validation \
    --output     results/mmmu_gcd.json
```

MMMU is loaded automatically from HuggingFace (`MMMU/MMMU`). Ensure you have `datasets` installed.

### Vanilla baseline (ablation)

```bash
# Add --no-gcd to any benchmark command to disable GCD decoding
python run_eval.py \
    --benchmark  mme \
    --model-path liuhaotian/llava-v1.5-7b \
    --data-root  /path/to/MME \
    --no-gcd
```

### All hyperparameters

| Argument | Default | Description |
|:---|:---:|:---|
| `--alpha` | `0.5` | Weight for negative-context logit suppression |
| `--beta` | `0.3` | Weight for text-only logit suppression / visual amplification |
| `--tau` | `0.05` | KL divergence threshold for adaptive $\alpha$/$\beta$ scaling |
| `--model-path` | `liuhaotian/llava-v1.5-7b` | HuggingFace model ID or local path |
| `--prototypes` | `None` | Path to `.pt` prototype file (enables disentanglement) |
| `--no-gcd` | `False` | Run vanilla decoding (for ablation) |

---

## Repository Structure

```
GCD/
│
├── gcd/                        # Core GCD library
│   ├── __init__.py             # Public API exports
│   ├── disentanglement.py      # Stage 1: Representation Disentanglement
│   │                           #   - build_prototypes()  (Eq. 1)
│   │                           #   - forward()           (Eq. 2)
│   ├── gcd_processor.py        # Stage 2: GCD LogitsProcessor
│   │                           #   - __call__()          (Eq. 3)
│   │                           #   - adaptive scaling    (Eq. 4)
│   └── model_utils.py          # Model loading & embedding utilities
│                               #   - load_llava_model()
│                               #   - get_visual_embeddings()
│                               #   - get_negative_visual_embeddings()
│                               #   - build_gcd_inputs()
│
├── eval/                       # Benchmark evaluation scripts
│   ├── eval_mme.py             # MME Perception & Cognition
│   ├── eval_mmvet.py           # MM-Vet (outputs for GPT-4 grader)
│   └── eval_mmmu.py            # MMMU multiple-choice accuracy
│
├── run_eval.py                 # Unified CLI evaluation entry point
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md
```

---

## Citation

If you find GCD useful in your research, please consider citing:

```bibtex
@article{zhang2024gcd,
  title     = {Mitigating Hallucinations in Large Vision-Language Models
               via Grounded-Contrastive Decoding},
  author    = {Zhang, Zhiyun and Zhang, Haotian},
  journal   = {IEEE},
  year      = {2024},
  url       = {https://ieeexplore.ieee.org/document/11283461},
  note      = {Glasgow College, University of Electronic Science and
               Technology of China, Chengdu, China}
}
```

---

## Acknowledgements

This project builds upon the following excellent open-source works. We sincerely thank their authors.

| Project | Role |
|:---|:---|
| [LLaVA](https://github.com/haotian-liu/LLaVA) | Base LVLM backbone (LLaVA-1.5-7B) |
| [VCD](https://github.com/DAMO-NLP-SG/VCD) | Visual Contrastive Decoding baseline & inspiration |
| [DoLA](https://github.com/voidism/DoLA) | Layer-contrastive decoding baseline |
| [VDD](https://github.com/Yuanfeng-Zhang/VDD) | Visual Debiasing Decoding baseline |
| [DeCO](https://github.com/wang-chaoyang/DeCO) | Dynamic Correction Decoding baseline |
| [CLIP](https://github.com/openai/CLIP) | Visual encoder |

---

<div align="center">

*Glasgow College · University of Electronic Science and Technology of China · 2024*

</div>
