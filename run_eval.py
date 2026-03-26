"""
Unified evaluation entry point for Grounded-Contrastive Decoding (GCD).

Supports three benchmarks:
  mme    – MME Perception & Cognition
  mmvet  – MM-Vet (outputs for GPT-4 grader)
  mmmu   – MMMU multiple-choice accuracy

Usage examples::

    # MME with GCD
    python run_eval.py --benchmark mme \
        --model-path liuhaotian/llava-v1.5-7b \
        --data-root /data/MME

    # MM-Vet without GCD (vanilla baseline)
    python run_eval.py --benchmark mmvet \
        --model-path liuhaotian/llava-v1.5-7b \
        --data-root /data/mm-vet \
        --no-gcd

    # MMMU with custom hyperparameters
    python run_eval.py --benchmark mmmu \
        --model-path liuhaotian/llava-v1.5-7b \
        --alpha 0.5 --beta 0.3 --tau 0.05

    # Use precomputed prototypes for disentanglement
    python run_eval.py --benchmark mme \
        --model-path liuhaotian/llava-v1.5-7b \
        --data-root /data/MME \
        --prototypes prototypes/coco_prototypes.pt
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Run GCD evaluation on MME / MM-Vet / MMMU benchmarks."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["mme", "mmvet", "mmmu"],
        help="Which benchmark to evaluate.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="liuhaotian/llava-v1.5-7b",
        help="HuggingFace model ID or local path to LLaVA model.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory of the benchmark dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results. Defaults to results/<benchmark>_gcd.json.",
    )
    parser.add_argument(
        "--prototypes",
        type=str,
        default=None,
        help="Path to precomputed confusion prototype file (.pt). "
             "If omitted, disentanglement is skipped.",
    )
    # GCD hyperparameters
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Negative-context suppression weight (default: 0.5).")
    parser.add_argument("--beta",  type=float, default=0.3,
                        help="Text-only suppression weight (default: 0.3).")
    parser.add_argument("--tau",   type=float, default=0.05,
                        help="KL divergence adaptive-scaling threshold (default: 0.05).")
    parser.add_argument("--no-gcd", action="store_true",
                        help="Disable GCD; run vanilla decoding for ablation.")
    # MMMU-specific
    parser.add_argument("--split", type=str, default="validation",
                        choices=["validation", "test"],
                        help="MMMU dataset split (default: validation).")

    args = parser.parse_args()

    # Default output paths
    if args.output is None:
        suffix = "vanilla" if args.no_gcd else "gcd"
        args.output = f"results/{args.benchmark}_{suffix}.json"

    print(f"\n{'='*60}")
    print(f"  Benchmark : {args.benchmark.upper()}")
    print(f"  Model     : {args.model_path}")
    print(f"  Mode      : {'Vanilla' if args.no_gcd else 'GCD'}")
    if not args.no_gcd:
        print(f"  alpha={args.alpha}  beta={args.beta}  tau={args.tau}")
    print(f"{'='*60}\n")

    # ---- Dispatch to benchmark-specific script ------------------------------
    if args.benchmark == "mme":
        if args.data_root is None:
            parser.error("--data-root is required for MME benchmark.")
        from eval.eval_mme import run_mme
        args.mme_root = args.data_root
        run_mme(args)

    elif args.benchmark == "mmvet":
        if args.data_root is None:
            parser.error("--data-root is required for MM-Vet benchmark.")
        from eval.eval_mmvet import run_mmvet
        args.mmvet_root = args.data_root
        run_mmvet(args)

    elif args.benchmark == "mmmu":
        from eval.eval_mmmu import run_mmmu
        run_mmmu(args)


if __name__ == "__main__":
    main()
