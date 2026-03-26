"""
Utility functions for model loading and visual embedding extraction.
Compatible with LLaVA-1.5-7B (CLIP ViT-L/14 + Vicuna/LLaMA-2-7B).
"""
import torch
from PIL import Image
from typing import Tuple


def load_llava_model(
    model_path: str,
    device: str = "cuda",
    load_8bit: bool = False,
    load_4bit: bool = False,
):
    """
    Load a LLaVA-1.5 model, tokenizer and image processor.

    Args:
        model_path: HuggingFace model ID or local path.
                    e.g. "liuhaotian/llava-v1.5-7b"
        device:     Target device string ("cuda", "cpu", etc.).
        load_8bit:  Enable 8-bit quantisation (bitsandbytes).
        load_4bit:  Enable 4-bit quantisation (bitsandbytes).

    Returns:
        tokenizer, model, image_processor, context_len
    """
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        model_base=None,
        model_name=model_name,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        device_map=device,
    )
    model.eval()
    return tokenizer, model, image_processor, context_len


# ---------------------------------------------------------------------------
# Visual feature extraction
# ---------------------------------------------------------------------------

def get_visual_embeddings(
    model,
    image_tensors: torch.Tensor,
) -> torch.Tensor:
    """
    Extract *projected* visual embeddings:
        f_v  →  CLIP encoder  →  MLP projector  →  projected_features

    Args:
        model:         LLaVA model instance.
        image_tensors: Preprocessed image tensor [B, C, H, W] (fp16/fp32).

    Returns:
        Projected visual embeddings [B, N_patches, D_llm].
    """
    with torch.no_grad():
        # 1) CLIP visual encoder
        visual_tower   = model.get_visual_tower()
        image_features = visual_tower(image_tensors)                    # [B, N, D_clip]

        # 2) MLP adapter / projector
        projected = model.get_model().mm_projector(image_features)      # [B, N, D_llm]
    return projected


def get_token_embeddings(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Return word-embedding table look-up for a token-id sequence."""
    return model.get_model().embed_tokens(input_ids)


# ---------------------------------------------------------------------------
# Negative-context visual features (noise perturbation, similar to VCD)
# ---------------------------------------------------------------------------

def get_negative_visual_embeddings(
    model,
    image_tensors: torch.Tensor,
    noise_std: float = 1.0,
) -> torch.Tensor:
    """
    Produce negative-context visual embeddings by adding Gaussian noise
    to the image tensor before encoding, encouraging the model to produce
    less visually-grounded outputs that can be contrasted away.

    Args:
        model:         LLaVA model instance.
        image_tensors: Original preprocessed image tensor [B, C, H, W].
        noise_std:     Standard deviation of additive Gaussian noise.

    Returns:
        Projected embeddings of the noisy image [B, N, D_llm].
    """
    with torch.no_grad():
        noise         = torch.randn_like(image_tensors) * noise_std
        noisy_images  = image_tensors + noise

        visual_tower  = model.get_visual_tower()
        noisy_features = visual_tower(noisy_images)
        projected_neg  = model.get_model().mm_projector(noisy_features)
    return projected_neg


# ---------------------------------------------------------------------------
# Full input-embedding construction
# ---------------------------------------------------------------------------

def build_gcd_inputs(
    model,
    tokenizer,
    image_processor,
    image: Image.Image,
    prompt: str,
    disentanglement_module=None,
    noise_std: float = 1.0,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct the three sets of input embeddings required by GCDLogitsProcessor:

      1. disentangled_embeds  – full sequence with disentangled visual tokens.
      2. negative_embeds      – full sequence with noise-perturbed visual tokens.
      3. text_only_embeds     – text tokens only, no visual tokens.

    Also returns the raw input_ids for bookkeeping.

    Args:
        model:                  LLaVA model instance.
        tokenizer:              Corresponding tokenizer.
        image_processor:        LLaVA image processor.
        image:                  PIL Image.
        prompt:                 Text prompt string (must include <image> token).
        disentanglement_module: Optional RepresentationDisentanglement instance.
                                When None, projected features are used as-is.
        noise_std:              Noise std for negative-context embeddings.
        device:                 Compute device.

    Returns:
        input_ids              [1, seq_len]       – text token IDs for reference
        disentangled_embeds    [1, merged_len, D] – main generation input
        negative_embeds        [1, merged_len, D]
        text_only_embeds       [1, text_len,   D]
    """
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX

    # ---- Image preprocessing ------------------------------------------------
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [t.to(device, dtype=torch.float16) for t in image_tensor]
    else:
        image_tensor = image_tensor.to(device, dtype=torch.float16)

    # ---- Tokenise prompt (with IMAGE_TOKEN_INDEX placeholder) ---------------
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    # ---- Projected visual features ------------------------------------------
    # Stack list → single tensor if necessary
    img_t = image_tensor if isinstance(image_tensor, torch.Tensor) \
            else torch.stack(image_tensor)

    projected_features = get_visual_embeddings(model, img_t)            # [1, N, D]
    projected_neg      = get_negative_visual_embeddings(
        model, img_t, noise_std=noise_std
    )                                                                     # [1, N, D]

    # ---- Optional disentanglement -------------------------------------------
    if disentanglement_module is not None:
        disentangled_features = disentanglement_module(projected_features)
    else:
        disentangled_features = projected_features

    # ---- Merge visual tokens into full embedding sequences ------------------
    disentangled_embeds = _merge_visual_into_embeds(
        model, input_ids, disentangled_features, device
    )
    negative_embeds = _merge_visual_into_embeds(
        model, input_ids, projected_neg, device
    )

    # ---- Text-only embeddings (no visual tokens) ----------------------------
    text_ids         = _strip_image_tokens(input_ids)
    text_only_embeds = get_token_embeddings(model, text_ids).to(
        device=device, dtype=torch.float16
    )

    return input_ids, disentangled_embeds, negative_embeds, text_only_embeds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _merge_visual_into_embeds(
    model,
    input_ids: torch.Tensor,
    visual_features: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """
    Replace the single IMAGE_TOKEN_INDEX placeholder in input_ids with
    visual_features and return the complete merged embedding sequence.

    Layout: [text_before_image | visual_patches | text_after_image]
    """
    from llava.constants import IMAGE_TOKEN_INDEX

    # Token embeddings for all positions (clamp avoids negative-index lookup
    # for IMAGE_TOKEN_INDEX = -200; those positions are replaced anyway)
    token_embeds = get_token_embeddings(model, input_ids.clamp(min=0))  # [1, L, D]
    token_embeds = token_embeds.to(device=device, dtype=torch.float16)

    # Locate image token position(s)
    image_mask = (input_ids[0] == IMAGE_TOKEN_INDEX)   # [L]
    if not image_mask.any():
        return token_embeds

    img_pos    = image_mask.nonzero(as_tuple=False)
    pre_end    = img_pos[0].item()
    post_start = img_pos[-1].item() + 1

    pre_image  = token_embeds[:, :pre_end, :]
    post_image = token_embeds[:, post_start:, :]

    merged = torch.cat([pre_image, visual_features, post_image], dim=1)
    return merged


def _strip_image_tokens(input_ids: torch.Tensor) -> torch.Tensor:
    """
    Remove IMAGE_TOKEN_INDEX entries from input_ids, keeping only text tokens.
    """
    from llava.constants import IMAGE_TOKEN_INDEX
    mask = input_ids[0] != IMAGE_TOKEN_INDEX
    return input_ids[:, mask]
