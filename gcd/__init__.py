"""
GCD – Grounded-Contrastive Decoding
====================================
A training-free framework that mitigates hallucinations in Large
Vision-Language Models (LVLMs) via:

  1. Representation Disentanglement (disentanglement.py)
     Removes spurious visual-text correlations using confusion prototypes
     and a lightweight cross-attention subtraction.

  2. Grounded-Contrastive Decoding (gcd_processor.py)
     Adjusts token probabilities at each step by contrasting disentangled
     visual logits against negative-context and text-only logits, with
     adaptive KL-divergence-based scaling.

Reference:
    "Mitigating Hallucinations in Large Vision-Language Models via
     Grounded-Contrastive Decoding"
    Zhiyun Zhang*, Haotian Zhang* – Glasgow College, UESTC, 2024.
"""

from .disentanglement import RepresentationDisentanglement
from .gcd_processor import GCDLogitsProcessor
from .model_utils import (
    load_llava_model,
    get_visual_embeddings,
    get_negative_visual_embeddings,
    get_token_embeddings,
    build_gcd_inputs,
)

__all__ = [
    "RepresentationDisentanglement",
    "GCDLogitsProcessor",
    "load_llava_model",
    "get_visual_embeddings",
    "get_negative_visual_embeddings",
    "get_token_embeddings",
    "build_gcd_inputs",
]
