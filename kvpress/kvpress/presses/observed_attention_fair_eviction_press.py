# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0


import logging
from dataclasses import dataclass
import torch
from torch import nn

from kvpress.presses.observed_attention_press import ObservedAttentionPress

logger = logging.getLogger(__name__)


@dataclass
class ObservedAttentionFairEvictionPress(ObservedAttentionPress):
    """
    Observed-attention with fair two-span eviction and local-queries fix for the earlier span.

    Differences vs parent:
      - When scoring keys in the earlier span, we only sum attention coming from
        queries within that earlier span (i.e., we zero out contributions from the later span).
      - Normalization divisor for those earlier-span keys is (earlier_end - k) instead of (L - k).
      - Kept indices are selected proportionally from extended earlier/later ranges:
            earlier_ext = [0 : earlier_end)
            later_ext   = [later_start : L)
        exactly like your SnapKVFairEvictionPress.

    Requires:
      - output_attentions=True and attn_implementation="eager" upstream.
      - `defense_span` and `sys_instr_span` set to adjacent, non-overlapping [start, end) pairs.
    """

    defense_span: tuple[int, int] | None = None
    sys_instr_span: tuple[int, int] | None = None

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        # attentions: (B, num_heads, L, L) with last two dims = (queries, keys)
        assert attentions is not None, (
            'Set output_attentions=True and attn_implementation="eager"'
        )
        bsz, num_kv_heads, L, _ = keys.shape

        # Validate spans and determine earlier/later
        assert self.defense_span is not None and self.sys_instr_span is not None, (
            "defense_span and sys_instr_span must be provided"
        )
        d0, d1 = self.defense_span
        s0, s1 = self.sys_instr_span
        for s, e, name in [(d0, d1, "defense_span"), (s0, s1, "sys_instr_span")]:
            assert 0 <= s <= e <= L, f"Invalid {name} {s, e} for L={L}"
        assert d1 == s0 or s1 == d0, "Spans must be adjacent and non-overlapping"

        if d1 <= s0:
            earlier_start, earlier_end = d0, d1
            later_start, later_end = s0, s1
        else:
            earlier_start, earlier_end = s0, s1
            later_start, later_end = d0, d1

        # Clone to avoid mutating upstream tensors
        att = attentions

        # ---- Local-queries fix for earlier span ----
        # We want keys in [earlier_start:earlier_end) to only receive attention mass
        # from queries in [earlier_start:earlier_end). That means zero out (q in later, k in earlier).
        if earlier_end > earlier_start and later_end > later_start:
            # Build a (L, L) mask over (queries, keys)
            mask = att.new_ones((L, L))
            mask[later_start:later_end, earlier_start:earlier_end] = 0.0
            # Broadcast over batch and heads: (B, H, L, L) * (1, 1, L, L)
            att = att * mask.view(1, 1, L, L)

        # Sum over queries dim to get per-key totals
        # scores_per_head: (B, num_heads, L)
        scores_per_head = att.sum(dim=2)

        # ---- Normalization by number of (allowed) future queries ----
        # For keys in earlier span: count = earlier_end - k
        # For keys in later span (and beyond): count = L - k
        ar = torch.arange(L, device=scores_per_head.device, dtype=scores_per_head.dtype)
        counts = L - ar  # default
        if earlier_end > earlier_start:
            # piecewise replace for earlier region
            counts_earlier = (earlier_end - ar).clamp_min(1)
            mask_earlier_pos = (ar >= earlier_start) & (ar < earlier_end)
            counts = torch.where(mask_earlier_pos, counts_earlier, counts)
        counts = counts.clamp_min(1)

        scores_per_head = scores_per_head / counts.view(1, 1, L)

        # ---- Average over KV groups to get (B, num_kv_heads, L) ----
        num_heads = scores_per_head.size(1)
        assert num_heads % num_kv_heads == 0, "Head grouping mismatch"
        num_kv_groups = num_heads // num_kv_heads
        scores = scores_per_head.view(bsz, num_kv_heads, num_kv_groups, L).mean(2)

        return scores

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.compression_ratio == 0:
            return keys, values

        scores = self.score(
            module, hidden_states, keys, values, attentions, kwargs
        )  # (B, kv_heads, L)
        B, H, L = scores.shape
        n_kept = int(L * (1 - self.compression_ratio))

        assert self.defense_span is not None and self.sys_instr_span is not None
        d0, d1 = self.defense_span
        s0, s1 = self.sys_instr_span
        assert 0 <= d0 <= d1 <= L and 0 <= s0 <= s1 <= L, "Invalid spans"
        assert d1 == s0 or s1 == d0, "Spans must be adjacent"

        if d1 <= s0:
            earlier_end = d1
            later_start = s0
        else:
            earlier_end = s1
            later_start = d0

        # Extended disjoint cover of [0, L)
        earlier_ext_start, earlier_ext_end = 0, earlier_end
        later_ext_start, later_ext_end = later_start, L

        len_earlier = max(0, earlier_ext_end - earlier_ext_start)
        len_later = max(0, later_ext_end - later_ext_start)
        assert len_earlier + len_later == L

        kept_earlier = (n_kept * len_earlier) // L
        kept_later = n_kept - kept_earlier

        if kept_earlier > 0 and len_earlier > 0:
            earlier_local = scores[:, :, earlier_ext_start:earlier_ext_end]
            earlier_idx = (
                earlier_local.topk(kept_earlier, dim=-1).indices + earlier_ext_start
            )
        else:
            earlier_idx = scores.new_empty((B, H, 0), dtype=torch.long)

        if kept_later > 0 and len_later > 0:
            later_local = scores[:, :, later_ext_start:later_ext_end]
            later_idx = later_local.topk(kept_later, dim=-1).indices + later_ext_start
        else:
            later_idx = scores.new_empty((B, H, 0), dtype=torch.long)

        indices = torch.cat([earlier_idx, later_idx], dim=-1)  # (B, H, n_kept)

        # Save raw per-position scores and kept indices for analysis
        try:
            self.position_scores_by_layer[module.layer_idx] = (  # type: ignore
                scores.detach().cpu()
            )  # (B, H, L)
            self.kept_indices_by_layer[module.layer_idx] = (  # type: ignore
                indices.detach().cpu()
            )  # (B, H, L') where L' is the pruned length
        except Exception:
            pass

        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)  # type: ignore

        # Prune keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        return keys, values
