# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import torch
from torch import nn

from kvpress.presses.tova_press import TOVAPress


@dataclass
class TOVAFairEvictionPress(TOVAPress):
    """
    TOVA-style scoring with fair two-span eviction.
    """

    defense_span: tuple[int, int] | None = None  # [start, end)
    sys_instr_span: tuple[int, int] | None = None  # [start, end)

    def _scores_from_anchor(
        self,
        attentions: torch.Tensor,  # (B, H, L, L)
        num_kv_heads: int,
        key_start: int,  # inclusive
        key_end: int,  # exclusive (must be <= anchor)
        anchor_idx: int,  # single query position
    ) -> torch.Tensor:
        """
        TOVA-style per-span scoring using ONE anchor token.
        Slice: queries = [anchor], keys = [key_start : key_end).
        Post-process like TOVA: mean over heads, then repeat to num_kv_heads.
        Returns (B, num_kv_heads, key_len).
        """
        bsz = attentions.size(0)
        key_len = max(0, key_end - key_start)
        if key_len <= 0:
            return attentions.new_zeros((bsz, num_kv_heads, 0))

        # (B, H, 1, key_len)
        attn_win = attentions[..., anchor_idx : anchor_idx + 1, key_start:key_end]
        scores = attn_win.mean(dim=1)
        scores = scores.repeat(1, num_kv_heads, 1)
        return scores

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        if attentions is None:
            raise ValueError(
                "Attentions are required for TOVAFairEvictionPress. "
                "Set output_attentions=True and attn_implementation='eager'."
            )

        assert self.defense_span is not None and self.sys_instr_span is not None
        bsz, num_kv_heads, q_len, _ = keys.shape

        d0, d1 = self.defense_span
        s0, s1 = self.sys_instr_span
        for s, e, name in [(d0, d1, "defense_span"), (s0, s1, "sys_instr_span")]:
            assert 0 <= s <= e <= q_len, f"Invalid {name} {s, e} for q_len={q_len}"

        if d1 == s0:
            defense_first = True
        elif s1 == d0:
            defense_first = False
        else:
            raise AssertionError("Spans must be adjacent and non-overlapping")

        if defense_first:
            earlier_span_start, earlier_span_end = d0, d1
            later_span_start, _ = s0, s1
        else:
            earlier_span_start, earlier_span_end = s0, s1
            later_span_start, _ = d0, d1

        # Extend to ends so ranges are disjoint and cover [0, L)
        #   earlier_ext = [0 : earlier_end)
        #   later_ext   = [later_start : L)
        earlier_ext_start, _ = 0, earlier_span_end
        later_ext_start, later_ext_end = later_span_start, q_len

        # Anchors = last token of each original span
        earlier_anchor = earlier_span_end - 1
        later_anchor = later_ext_end - 1

        scores = keys[..., 0].new_zeros((bsz, num_kv_heads, q_len))

        # Keys voted on: earlier_ext before the anchor
        earlier_keys_start = earlier_ext_start
        earlier_keys_end = earlier_anchor
        # Compute TOVA-style scores from single anchor to keys in its span
        scores_earlier = self._scores_from_anchor(
            attentions=attentions,
            num_kv_heads=num_kv_heads,
            key_start=earlier_keys_start,
            key_end=earlier_keys_end,
            anchor_idx=earlier_anchor,
        )
        # Write into the global tensor
        if scores_earlier.numel() > 0:
            scores[:, :, earlier_keys_start:earlier_keys_end] = scores_earlier

        # Protect the earlier anchor by setting it to this span's max
        if earlier_span_end > earlier_span_start:
            if scores_earlier.numel() > 0:
                earlier_keep = torch.amax(
                    scores_earlier, dim=-1, keepdim=True
                )  # (B, kv_heads, 1)
            else:
                earlier_keep = torch.ones(
                    (bsz, num_kv_heads, 1), device=scores.device, dtype=scores.dtype
                )
            scores[:, :, earlier_anchor : earlier_anchor + 1] = earlier_keep

        # Later span pass
        later_keys_start = later_ext_start
        scores_later = self._scores_from_anchor(
            attentions=attentions,
            num_kv_heads=num_kv_heads,
            key_start=later_keys_start,
            key_end=later_anchor,
            anchor_idx=later_anchor,
        )
        if scores_later.numel() > 0:
            scores[:, :, later_keys_start:later_anchor] = scores_later

        # protect the last token
        keep = (
            scores_later.amax(dim=-1, keepdim=True)
            if scores_later.numel() > 0
            else torch.ones(
                (bsz, num_kv_heads, 1), device=scores.device, dtype=scores.dtype
            )
        )
        scores[:, :, later_anchor : later_anchor + 1] = keep

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

        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        B, H, L = scores.shape
        n_kept = int(L * (1 - self.compression_ratio))

        # Validate spans and figure out earlier/later
        assert self.defense_span is not None and self.sys_instr_span is not None
        d0, d1 = self.defense_span
        s0, s1 = self.sys_instr_span
        assert 0 <= d0 <= d1 <= L and 0 <= s0 <= s1 <= L, (
            f"Invalid spans: defense={self.defense_span}, sys={self.sys_instr_span}, L={L}"
        )
        assert d1 == s0 or s1 == d0, "Spans must be adjacent"

        # earlier/later by order
        if d1 <= s0:
            _, earlier_end = d0, d1
            later_start, _ = s0, s1
        else:
            _, earlier_end = s0, s1
            later_start, _ = d0, d1

        # Extend to ends so ranges are disjoint and cover [0, L)
        #    earlier_range = [0 : earlier_end)
        #    later_range   = [later_start : L)
        earlier_ext_start, earlier_ext_end = 0, earlier_end
        later_ext_start, later_ext_end = later_start, L

        len_earlier = max(0, earlier_ext_end - earlier_ext_start)
        len_later = max(0, later_ext_end - later_ext_start)
        assert len_earlier + len_later == L, (
            "Extended ranges must cover the whole sequence without overlap"
        )

        # Proportional quotas by extended lengths
        kept_earlier = (n_kept * len_earlier) // L
        kept_later = n_kept - kept_earlier

        # Top-k inside each extended range (handle zero-k cleanly)
        if kept_earlier > 0 and len_earlier > 0:
            earlier_local = scores[:, :, earlier_ext_start:earlier_ext_end]
            earlier_idx = earlier_local.topk(kept_earlier, dim=-1).indices  # (B,H,k_e)
            earlier_idx = earlier_idx + earlier_ext_start
        else:
            earlier_idx = scores.new_empty((B, H, 0), dtype=torch.long)

        if kept_later > 0 and len_later > 0:
            later_local = scores[:, :, later_ext_start:later_ext_end]
            later_idx = later_local.topk(kept_later, dim=-1).indices  # (B,H,k_l)
            later_idx = later_idx + later_ext_start
        else:
            later_idx = scores.new_empty((B, H, 0), dtype=torch.long)

        indices = torch.cat([earlier_idx, later_idx], dim=-1)  # (B,H,n_kept)

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
