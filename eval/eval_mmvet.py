"""
MM-Vet Benchmark Evaluation with GCD.

MM-Vet evaluates integrated multimodal capabilities across six skills:
  Rec  – recognition
  OCR  – optical character recognition
  Know – knowledge
  Gen  – language generation
  Spat – spatial awareness
  Math – mathematical reasoning

The official scorer uses GPT-4 to grade free-form answers.
This script generates model outputs; run the official MM-Vet grader
afterwards to obtain final scores.

Expected dataset layout::

    <mmvet-root>/
      ├── images/           (*.jpg / *.png)
      └── mm-vet.json       (official question file)

mm-vet.json structure::

    {
      "v1_0": {
        "imagename": "v1_0.png",
        "question":  "...",
        "answer":    "...",
        "capability": ["rec", "ocr"]
      },
      ...
    }

Usage::

    python eval/eval_mmvet.py \
        --model-path liuhaotian/llava-v1.5-7b \
        --mmvet-root /data/mm-vet \
        --output     results/mmvet_gcd.json \
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_prompt(question: str) -> str:
    return (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        f"USER: <image>\n{question}\nASSISTANT:"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_mmvet(args):
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

    # ---- Load questions -----------------------------------------------------
    qfile   = os.path.join(args.mmvet_root, "mm-vet.json")
    img_dir = os.path.join(args.mmvet_root, "images")

    with open(qfile) as f:
        questions = json.load(f)

    outputs: dict = {}

    for qid, item in tqdm(questions.items(), desc="MM-Vet"):
        image_path = os.path.join(img_dir, item["imagename"])
        image      = Image.open(image_path).convert("RGB")
        prompt     = build_prompt(item["question"])

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
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=0,
                )
                # Vanilla: output_ids = [prompt_tokens | generated_tokens]
                generated = tokenizer.decode(
                    output_ids[0, input_ids.shape[1]:],
                    skip_special_tokens=True,
                ).strip()

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
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=0,
                    logits_processor=LogitsProcessorList([gcd_processor]),
                )
                # GCD: generate(inputs_embeds=...) returns only newly generated
                # token IDs (no prompt prefix in output tensor)
                generated = tokenizer.decode(
                    output_ids[0],
                    skip_special_tokens=True,
                ).strip()

        outputs[qid] = generated

    # ---- Save outputs for official MM-Vet grader ----------------------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated answers saved to {args.output}")
    print("Run the official MM-Vet GPT-4 grader on this file to obtain final scores.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--mmvet-root", type=str, required=True,
                        help="Root directory of the MM-Vet dataset.")
    parser.add_argument("--output",     type=str, default="results/mmvet_gcd.json")
    parser.add_argument("--prototypes", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta",  type=float, default=0.3)
    parser.add_argument("--tau",   type=float, default=0.05)
    parser.add_argument("--no-gcd", action="store_true")
    args = parser.parse_args()
    run_mmvet(args)
