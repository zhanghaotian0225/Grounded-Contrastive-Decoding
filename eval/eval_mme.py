"""
MME Benchmark Evaluation with GCD.

MME measures multimodal perception across 14 sub-tasks
(Existence, Count, Position, Color, Posters, Celebrity, Scene,
 Landmark, Artwork, OCR, Commonsense Reasoning, Numerical Calculation,
 Text Translation, Code Reasoning).

Each question is a Yes/No question; accuracy per sub-task is reported
along with a total Perception / Cognition score.

Expected dataset layout::

    <mme_root>/
      ├── Existence/
      │     ├── images/   (*.jpg / *.png)
      │     └── questions.jsonl
      ├── Count/
      ...

Each line in questions.jsonl::

    {
      "image": "00001.jpg",
      "question": "Is there a cat in the image?",
      "answer": "Yes"
    }

Usage::

    python eval/eval_mme.py \
        --model-path liuhaotian/llava-v1.5-7b \
        --mme-root   /data/MME \
        --output     results/mme_gcd.json \
        [--alpha 0.5] [--beta 0.3] [--tau 0.05] [--no-gcd]
"""

import argparse
import json
import os

import torch
from PIL import Image
from tqdm import tqdm
from transformers import LogitsProcessorList

from gcd import (
    GCDLogitsProcessor,
    RepresentationDisentanglement,
    build_gcd_inputs,
    load_llava_model,
)

# Sub-tasks that contribute to the Perception score
PERCEPTION_TASKS = [
    "Existence", "Count", "Position", "Color",
    "Posters", "Celebrity", "Scene", "Landmark", "Artwork", "OCR",
]

# Sub-tasks that contribute to the Cognition score
COGNITION_TASKS = [
    "commonsense_reasoning", "numerical_calculation",
    "text_translation", "code_reasoning",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_prompt(question: str) -> str:
    """Wrap the question for LLaVA's instruction format."""
    return (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        f"USER: <image>\n{question}\nPlease answer yes or no. ASSISTANT:"
    )


def extract_yes_no(text: str) -> str:
    """Return 'Yes', 'No', or 'Unknown' from model output."""
    text = text.strip().lower()
    if text.startswith("yes"):
        return "Yes"
    if text.startswith("no"):
        return "No"
    return "Unknown"


def score_task(predictions: list) -> float:
    """
    MME scoring: each correct answer scores 2 points (max 200 per task).
    """
    if not predictions:
        return 0.0
    correct = sum(1 for p in predictions if p["pred"] == p["gt"])
    return correct / len(predictions) * 200.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_mme(args):
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
        dis_module.load_prototypes(args.prototypes, device="cuda")
        dis_module.eval()

    # ---- Collect task folders -----------------------------------------------
    task_dirs = sorted(
        d for d in os.listdir(args.mme_root)
        if os.path.isdir(os.path.join(args.mme_root, d))
    )

    all_results: dict = {}
    task_scores: dict = {}

    for task_name in task_dirs:
        task_dir = os.path.join(args.mme_root, task_name)
        img_dir  = os.path.join(task_dir, "images")
        qfile    = os.path.join(task_dir, "questions.jsonl")

        if not os.path.isfile(qfile):
            continue

        with open(qfile) as f:
            questions = [json.loads(line) for line in f]

        predictions: list = []

        for item in tqdm(questions, desc=task_name):
            image_path = os.path.join(img_dir, item["image"])
            image      = Image.open(image_path).convert("RGB")
            prompt     = build_prompt(item["question"])
            gt_answer  = item["answer"]

            with torch.inference_mode():
                if args.no_gcd:
                    # ---- Vanilla decoding ------------------------------------
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
                        max_new_tokens=10,
                        do_sample=False,
                        temperature=0,
                    )
                    # Vanilla: output_ids = [prompt_tokens | generated_tokens]
                    generated = tokenizer.decode(
                        output_ids[0, input_ids.shape[1]:],
                        skip_special_tokens=True,
                    )

                else:
                    # ---- GCD decoding ----------------------------------------
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
                        max_new_tokens=10,
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

            pred = extract_yes_no(generated)
            predictions.append({"image": item["image"], "pred": pred, "gt": gt_answer})

        task_scores[task_name] = score_task(predictions)
        all_results[task_name] = predictions
        print(f"  [{task_name}] score = {task_scores[task_name]:.2f}")

    # ---- Aggregate ----------------------------------------------------------
    perception = sum(v for k, v in task_scores.items() if k in PERCEPTION_TASKS)
    cognition  = sum(v for k, v in task_scores.items() if k in COGNITION_TASKS)
    total      = perception + cognition

    print("\n" + "=" * 50)
    print(f"  Perception : {perception:.2f}")
    print(f"  Cognition  : {cognition:.2f}")
    print(f"  Total      : {total:.2f}")
    print("=" * 50)

    # ---- Save results -------------------------------------------------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "scores":     task_scores,
                "perception": perception,
                "cognition":  cognition,
                "total":      total,
                "details":    all_results,
            },
            f, indent=2,
        )
    print(f"Results saved to {args.output}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--mme-root",   type=str, required=True,
                        help="Root directory of the MME dataset.")
    parser.add_argument("--output",     type=str, default="results/mme_gcd.json")
    parser.add_argument("--prototypes", type=str, default=None,
                        help="Path to precomputed prototype file (.pt).")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta",  type=float, default=0.3)
    parser.add_argument("--tau",   type=float, default=0.05)
    parser.add_argument("--no-gcd", action="store_true",
                        help="Vanilla decoding (no GCD) for ablation.")
    args = parser.parse_args()
    run_mme(args)
