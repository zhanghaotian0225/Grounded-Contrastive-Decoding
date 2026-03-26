"""
Grounded-Contrastive Decoding (GCD) LogitsProcessor.

Implements Equations (3) and (4) from the paper:

  (3)  s_GCD(y_t) = (1 + beta) * l_v(y_t)
                  -  alpha      * l_neg(y_t)
                  -  beta       * l_text(y_t)

  (4)  Adaptive scaling at each step:
         if KL(p_GCD || p_v) > tau:
             alpha <- alpha * tau / KL(p_GCD || p_v)
             beta  <- beta  * tau / KL(p_GCD || p_v)
         else (KL <= tau):
             alpha <- min(alpha * tau / KL, alpha_init)   # slight recovery
             beta  <- min(beta  * tau / KL, beta_init)

Where:
  l_v(y_t)    – logits from the disentangled visual embedding  (main forward)
  l_neg(y_t)  – logits from the noise-perturbed visual embedding
  l_text(y_t) – logits from a text-only (no vision) run

The adaptive KL-scaling prevents over-suppression when KL > tau (damping),
and strengthens grounding when KL <= tau (slight recovery up to initial values).
"""

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor


class GCDLogitsProcessor(LogitsProcessor):
    """
    Plug-and-play HuggingFace LogitsProcessor that applies
    Grounded-Contrastive Decoding at every autoregressive step.

    Parameters α (alpha), β (beta), and τ (tau) default to the values
    reported in the paper (α=0.5, β=0.3, τ=0.05).

    Example usage::

        processor = GCDLogitsProcessor(
            model=model,
            input_ids=input_ids,
            disentangled_inputs_embeds=dis_emb,
            negative_inputs_embeds=neg_emb,
            text_inputs_embeds=txt_emb,
        )
        output = model.generate(
            inputs_embeds=dis_emb,
            logits_processor=LogitsProcessorList([processor]),
            ...
        )
    """

    def __init__(
        self,
        model,
        input_ids: torch.Tensor,
        disentangled_inputs_embeds: torch.Tensor,
        negative_inputs_embeds: torch.Tensor,
        text_inputs_embeds: torch.Tensor,
        alpha: float = 0.5,
        beta: float = 0.3,
        tau: float = 0.05,
    ):
        """
        Args:
            model:                      LLaVA model (used for auxiliary forward passes).
            input_ids:                  Original prompt token IDs [1, prompt_len].
            disentangled_inputs_embeds: Full embedding sequence (disentangled vision)
                                        [1, seq_len, D].
            negative_inputs_embeds:     Full embedding sequence (noisy/negative vision)
                                        [1, seq_len, D].
            text_inputs_embeds:         Text-only embedding sequence [1, text_len, D].
            alpha:                      Suppression weight for negative-context logits.
            beta:                       Suppression weight for text-only logits.
            tau:                        KL divergence threshold for adaptive scaling.
        """
        self.model = model

        # Store the three sets of base embeddings
        self.dis_embeds = disentangled_inputs_embeds
        self.neg_embeds = negative_inputs_embeds
        self.txt_embeds = text_inputs_embeds

        # Hyperparameters – mutable across steps by adaptive scaling
        self.alpha = alpha
        self.beta  = beta
        self.tau   = tau

        # Store initial values as a recovery ceiling (prevent unbounded growth)
        self._alpha_init = alpha
        self._beta_init  = beta

        # KV-cache for the two auxiliary forward passes
        self._pkv_neg = None
        self._pkv_txt = None
        self._step    = 0

    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GCD-adjusted logits.

        Called automatically by HuggingFace generate() at each step.

        Args:
            input_ids: All token IDs generated so far [B, current_len].
            scores:    Raw logits from the main (disentangled) forward pass
                       [B, vocab_size].  These are l_v.

        Returns:
            GCD-adjusted logits [B, vocab_size].
        """
        l_v = scores  # [B, V]

        # Compute auxiliary logits
        l_neg  = self._forward_aux(input_ids, self.neg_embeds, cache_key="neg")
        l_text = self._forward_aux(input_ids, self.txt_embeds, cache_key="txt")

        # ---- Equation (3): GCD score ----------------------------------------
        s_gcd = (1.0 + self.beta) * l_v \
              - self.alpha         * l_neg \
              - self.beta          * l_text

        # ---- Equation (4): Adaptive KL-based scaling ------------------------
        p_v   = F.softmax(l_v,   dim=-1)   # original visual distribution
        p_gcd = F.softmax(s_gcd, dim=-1)   # GCD distribution

        # KL(p_GCD || p_v) = sum(p_GCD * log(p_GCD / p_v))
        # F.kl_div(input, target) computes sum(target * (log(target) - input))
        # So: F.kl_div(p_v.log(), p_gcd) = sum(p_gcd * (log(p_gcd) - log(p_v)))
        #                                 = KL(p_gcd || p_v)  ✓
        kl = F.kl_div(
            (p_v + 1e-10).log(),
            p_gcd,
            reduction="batchmean",
        ).item()

        if kl > self.tau:
            # KL too large → dampen alpha and beta to reduce aggressiveness
            scale      = self.tau / (kl + 1e-10)
            self.alpha = self.alpha * scale
            self.beta  = self.beta  * scale
        else:
            # KL within range → slightly recover alpha and beta toward initial values
            # This "strengthens grounding" as described in the paper
            if kl > 1e-10:
                recover    = min(self.tau / kl, 1.05)  # cap single-step growth at 5%
                self.alpha = min(self.alpha * recover, self._alpha_init)
                self.beta  = min(self.beta  * recover, self._beta_init)

        # Recompute with (possibly updated) alpha and beta
        s_gcd = (1.0 + self.beta) * l_v \
              - self.alpha         * l_neg \
              - self.beta          * l_text

        self._step += 1
        return s_gcd

    # ------------------------------------------------------------------
    # Auxiliary forward pass with KV-caching
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward_aux(
        self,
        input_ids: torch.Tensor,
        base_embeds: torch.Tensor,
        cache_key: str,
    ) -> torch.Tensor:
        """
        Single forward pass for an auxiliary context (negative or text-only).
        Uses KV-caching so only the newest token is re-evaluated after the
        first step, keeping latency overhead small.

        Args:
            input_ids:   Current full token sequence [B, cur_len].
            base_embeds: Base embedding sequence for the prompt part [1, L, D].
            cache_key:   "neg" or "txt" – selects which KV-cache to use.

        Returns:
            Logits for the next token [B, vocab_size].
        """
        pkv_attr = f"_pkv_{cache_key}"
        past_kv  = getattr(self, pkv_attr)

        if past_kv is None:
            # First decoding step: full forward on prompt embeddings
            out = self.model(
                inputs_embeds=base_embeds,
                use_cache=True,
                return_dict=True,
            )
        else:
            # Subsequent steps: embed only the most-recent generated token
            last_token_id = input_ids[:, -1:]                      # [B, 1]
            last_embed    = self.model.get_model().embed_tokens(    # [B, 1, D]
                last_token_id
            ).to(dtype=base_embeds.dtype)

            out = self.model(
                inputs_embeds=last_embed,
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )

        # Update KV-cache
        setattr(self, pkv_attr, out.past_key_values)

        return out.logits[:, -1, :]  # [B, vocab_size]
