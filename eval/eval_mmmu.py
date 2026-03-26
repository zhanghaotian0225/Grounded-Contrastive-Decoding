"""
MMMU Benchmark Evaluation with GCD.

MMMU (Massive Multi-discipline Multimodal Understanding and Reasoning)
covers 30 subjects across 6 disciplines with multiple-choice questions
requiring expert-level knowledge and cross-modal reasoning.

This script uses the HuggingFace datasets version of MMMU.

Dataset field reference (MMMU/MMMU on HuggingFace):
  - "question"   : question text
  - "options"    : list of option strings, e.g. ["A. ...", "B. ...", ...]
                   (or a JSON string representation of that list)
  - "answer"     : correct option letter, e.g. "A"
  - "image_1"    : PIL Image (primary image; up to image_7 may exist)
  - "subject"    : subject name, e.g. "Accounting"

Usage::

    python eval/eval_mmmu.py \
        --model-path  liuhaotian/llava-v1.5-7b \
        --split       validation \
        --output      results/mmmu_gcd.json \
        [--alpha 0.5] [--beta 0.3] [--tau 0.05] [--no-gcd]

The script reports overall accuracy and per-subject accuracy.
"""

import argparse
import ast
import json
import os
import re
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import LogitsProcessorList

from gcd import (
    GCDLogitsProcessor,
    RepresentationDisentanglement,
    build_gcd_inputs,
    load_llava_model,
)

# Valid answer letters
OPTIONS = ["A", "B", "C", "D", "E"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_options(raw) -> list:
    """
    Parse the 'options' field from an MMMU item.

    The HuggingFace MMMU dataset stores options as either:
      - a Python list:  ["A. ...", "B. ...", ...]
      - a JSON/repr string: "['A. ...', 'B. ...', ...]"
    """
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            pass
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return []


def build_prompt(question: str, options: list) -> str:
    """Format a multiple-choice question for LLaVA."""
    lines = []
    for i, opt in enumerate(options):
        prefix = f"{chr(65 + i)}."
        # Add prefix only if the option doesn't already start with "A." / "B." etc.
        if opt.strip().startswith(prefix):
            lines.append(opt.strip())
        else:
            lines.append(f"{prefix} {opt.strip()}")
    opts_text = "\n".join(lines)
    return (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        f"USER: <image>\n{question}\n{opts_text}\n"
        "Answer with the option's letter from the given choices directly. ASSISTANT:"
    )


def extract_option(text: str) -> str:
    """Extract the predicted option letter (A–E) from model output."""
    text = text.strip()
    # Match a standalone capital letter A–E
    match = re.search(r"\b([A-E])\b", text)
    if match:
        return match.group(1)
    # Fallback: first character
    if text and text[0].upper() in OPTIONS:
        return text[0].upper()
    return "A"   # default to first option


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_mmmu(args):
    # ---- Load model ---------------------------------------------------------
    print("Loading model...")
    tokenizer, model, image_processor, _ = load_llava_model(
        args.model_path, device="cuda"
    )

    # ---- Optional disentanglement module ------------------------------------
    dis_module = None
    if not args.no_gcd and args.prototypes:
        embed_dim  = model.config.hidden_size
        dis_module = RepresentationDisentanglement(embed_dim=embed_dim).cuda()
        dis_module.load_prototypes(args.prototypes)
        dis_module.eval()

    # ---- Load MMMU dataset --------------------------------------------------
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install: pip install datasets")

    dataset = load_dataset(
        "MMMU/MMMU",
        "all",
        split=args.split,
        trust_remote_code=True,
    )

    all_preds: list = []
    subject_results: dict = defaultdict(list)

    for item in tqdm(dataset, desc="MMMU"):
        # Skip items without a primary image
        if item.get("image_1") is None:
            continue

        image   = item["image_1"].convert("RGB")
        subject = item.get("subject", "unknown")

        # Parse options – HuggingFace MMMU uses an "options" list/string field
        options = parse_options(item.get("options", []))
        if not options:
            continue

        prompt = build_prompt(item["question"], options)
        gt     = item["answer"]

        with torch.inference_mode():
            if args.no_gcd:
                # ---- Vanilla decoding ----------------------------------------
                from llava.mm_utils import process_images, tokenizer_image_token
                from llava.constants import IMAGE_TOKEN_INDEX

                image_tensor = process_images(
                    [image], image_processor, model.config
                ).to("cuda", dtype=torch.float16)

                input_ids = tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).unsqueeze(0).cuda()

                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    max_new_tokens=8,
                    do_sample=False,
                    temperature=0,
                )
                # Vanilla: output_ids = [prompt_tokens | generated_tokens]
                generated = tokenizer.decode(
                    output_ids[0, input_ids.shape[1]:],
                    skip_special_tokens=True,
                )

            else:
                # ---- GCD decoding --------------------------------------------
                input_ids, dis_emb, neg_emb, txt_emb = build_gcd_inputs(
                    model, tokenizer, image_processor,
                    image, prompt,
                    disentanglement_module=dis_module,
                    device="cuda",
                )

                gcd_processor = GCDLogitsProcessor(
                    model=model,
                    input_ids=input_ids,
                    disentangled_inputs_embeds=dis_emb,
                    negative_inputs_embeds=neg_emb,
                    text_inputs_embeds=txt_emb,
                    alpha=args.alpha,
                    beta=args.beta,
                    tau=args.tau,
                )

                output_ids = model.generate(
                    inputs_embeds=dis_emb,
                    max_new_tokens=8,
                    do_sample=False,
                    temperature=0,
                    logits_processor=LogitsProcessorList([gcd_processor]),
                )
                # GCD: generate(inputs_embeds=...) returns only newly generated
                # token IDs (no prompt prefix in output tensor)
                generated = tokenizer.decode(
                    output_ids[0],
                    skip_special_tokens=True,
                )

        pred    = extract_option(generated)
        correct = int(pred == gt)

        all_preds.append({"pred": pred, "gt": gt, "correct": correct})
        subject_results[subject].append(correct)

    # ---- Aggregate ----------------------------------------------------------
    overall_acc = sum(p["correct"] for p in all_preds) / len(all_preds) * 100
    per_subject = {
        subj: sum(v) / len(v) * 100
        for subj, v in subject_results.items()
    }

    print("\n" + "=" * 50)
    print(f"  Overall Accuracy: {overall_acc:.2f}%")
    print("  Per-subject:")
    for subj, acc in sorted(per_subject.items(), key=lambda x: -x[1]):
        print(f"    {subj:40s} {acc:.1f}%")
    print("=" * 50)

    # ---- Save ---------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "overall_accuracy": overall_acc,
                "per_subject":      per_subject,
                "predictions":      all_preds,
            },
            f, indent=2,
        )
    print(f"Results saved to {args.output}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--split",      type=str, default="validation",
                        choices=["validation", "test"])
    parser.add_argument("--output",     type=str, default="results/mmmu_gcd.json")
    parser.add_argument("--prototypes", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta",  type=float, default=0.3)
    parser.add_argument("--tau",   type=float, default=0.05)
    parser.add_argument("--no-gcd", action="store_true")
    args = parser.parse_args()
    run_mmmu(args)
