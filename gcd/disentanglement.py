"""
Representation Disentanglement Module for GCD.

Implements Equations (1) and (2) from the paper:

  (1)  D_v[c] = (1 / |S_c|) * sum_{i in S_c} OriginalProj(f_v^(i))

  (2)  v = OriginalProj(f_v) - Softmax( Q K^T / sqrt(d) ) * V
           where  Q = W_q * OriginalProj(f_v)
                  K = W_k * D_v^T
                  V = W_v * D_v^T

The module is lightweight (three linear projections) and requires only
precomputed prototypes at inference time, adding negligible overhead.

Training-free design: W_q, W_k, W_v are initialised as identity matrices
so that the module can run immediately without any training, while still
allowing optional fine-tuning if desired.
"""

import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RepresentationDisentanglement(nn.Module):
    """
    Removes spurious visual–language correlations by subtracting the
    contribution of confusion prototypes via a cross-attention mechanism.

    The projection matrices W_q, W_k, W_v are initialised as identity
    transformations so the module is immediately usable in a training-free
    setting.  They can be fine-tuned if labelled data is available.

    Usage::

        module = RepresentationDisentanglement(embed_dim=4096).cuda()
        module.build_prototypes(embeddings_by_category)   # offline
        # or
        module.load_prototypes("prototypes.pt")

        disentangled = module(projected_features)  # at inference
    """

    def __init__(self, embed_dim: int, num_heads: int = 8):
        """
        Args:
            embed_dim: Dimensionality of the projected visual embeddings
                       (must match the LLM hidden size, e.g. 4096 for 7B).
            num_heads: Number of attention heads (used for head_dim scaling).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads

        # Lightweight projection matrices (no bias, float32).
        # Initialised as identity so the module is immediately training-free:
        #   Q = I * f  ≡ f,  K = I * D  ≡ D,  V = I * D  ≡ D
        # All computation in forward() is promoted to float32 to avoid
        # dtype mismatches when projected_features arrives in float16.
        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self._init_identity()

        # Prototype dictionary – populated via build_prototypes / load_prototypes
        self.register_buffer("prototypes", None)

    def _init_identity(self):
        """Initialise all projection matrices as identity transformations."""
        nn.init.eye_(self.Wq.weight)
        nn.init.eye_(self.Wk.weight)
        nn.init.eye_(self.Wv.weight)

    # ------------------------------------------------------------------
    # Prototype management
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_prototypes(
        self,
        embeddings_by_category: Dict[str, List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute the confusion prototype dictionary from per-category embeddings.

        Implements Equation (1):
            D_v[c] = (1 / |S_c|) * sum_{i in S_c} OriginalProj(f_v^(i))

        Args:
            embeddings_by_category: Mapping from category name to a list of
                projected visual embeddings (each of shape [D] or [N_patches, D]).
                Representative confusing categories include common object
                co-occurrences, attribute confusions, and relational errors.

        Returns:
            Prototype tensor of shape [C, D] (also stored in self.prototypes).
        """
        proto_list: List[torch.Tensor] = []
        for category, embs in embeddings_by_category.items():
            if not embs:          # skip empty category lists
                continue
            # Each emb may be [D] or [N_patches, D] – average spatial patches
            stacked = []
            for e in embs:
                if e.dim() == 2:  # [N_patches, D] → average over patches
                    e = e.mean(dim=0)
                stacked.append(e.float())   # normalise to float32
            category_proto = torch.stack(stacked, dim=0).mean(dim=0)  # [D]
            proto_list.append(category_proto)

        if not proto_list:
            raise ValueError("embeddings_by_category contains no valid entries.")

        prototypes = torch.stack(proto_list, dim=0)  # [C, D]
        self.prototypes = prototypes                  # update registered buffer
        return prototypes

    def load_prototypes(self, path: str, device: Optional[str] = None):
        """
        Load precomputed prototypes from a .pt file saved by save_prototypes().

        Prototypes are automatically moved to the same device as the module
        (or to the explicitly supplied device) to avoid per-forward CPU→GPU copies.

        Args:
            path:   Path to the saved tensor file.
            device: Target device string. If None, inferred from the module's
                    current parameter device.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Prototype file not found: {path}")

        # Determine target device
        if device is None:
            try:
                device = str(next(self.parameters()).device)
            except StopIteration:
                device = "cpu"

        protos = torch.load(path, map_location=device)
        self.prototypes = protos.float()   # store in float32 to match Linear weights

    def save_prototypes(self, path: str):
        """Persist the current prototype dictionary to disk."""
        if self.prototypes is None:
            raise RuntimeError(
                "No prototypes to save – call build_prototypes() first."
            )
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.prototypes, path)
        print(f"Saved {self.prototypes.shape[0]} prototypes to {path}")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, projected_features: torch.Tensor) -> torch.Tensor:
        """
        Apply disentanglement to projected visual features.

        Implements Equation (2):
            v = OriginalProj(f_v) - Softmax( Q K^T / sqrt(d) ) * V

        All internal computation is performed in float32 to avoid dtype
        conflicts between the fp32 Linear weights and fp16 input features.
        The output is cast back to the original dtype of projected_features.

        If no prototypes have been loaded, the input is returned unchanged
        (graceful fallback so the rest of the pipeline remains unaffected).

        Args:
            projected_features: MLP-projected visual embeddings [B, N, D].

        Returns:
            Disentangled embeddings [B, N, D], same dtype as input.
        """
        if self.prototypes is None:
            return projected_features

        orig_dtype = projected_features.dtype
        device     = projected_features.device

        # Promote to float32 for stable computation (avoids fp16 / fp32 clash)
        features_f32 = projected_features.float()                        # [B, N, D]
        protos_f32   = self.prototypes.to(device=device).float()         # [C, D]

        # ---- Cross-attention: Q from input, K/V from prototypes --------------
        Q = self.Wq(features_f32)   # [B, N, D]
        K = self.Wk(protos_f32)     # [C, D]
        V = self.Wv(protos_f32)     # [C, D]

        # Scaled dot-product attention scores: [B, N, D] x [D, C] → [B, N, C]
        scale        = self.head_dim ** -0.5
        attn_scores  = torch.einsum("bnd,cd->bnc", Q, K) * scale
        attn_weights = F.softmax(attn_scores, dim=-1)     # [B, N, C]

        # Weighted sum of prototype values → spurious component [B, N, D]
        spurious = torch.einsum("bnc,cd->bnd", attn_weights, V)

        # Subtract spurious contribution (Equation 2), cast back to input dtype
        disentangled = (features_f32 - spurious).to(dtype=orig_dtype)
        return disentangled
