# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

from kvpress.presses.snapkv_press import SnapKVPress


@dataclass
class SnapKVFairEvictionPress(SnapKVPress):
    """
    SnapKV with fair two-window voting for defense + system-instruction prompts.

    Implementation:
      Two span-local windows, each "votes" on earlier tokens within its own span:
        - Earlier span window: last W_e tokens vote on keys in
            [earlier_span_start : earlier_span_end - W_e).
        - Later   span window: last W_l tokens vote on keys in
            [later_span_start   : later_span_end   - W_l).

      We split the global window_size into two halves:
        W_e = min(window_size // 2, len(earlier_span))
        W_l = min(window_size - W_e, len(later_span))

      Then, for each window:
        - Compute Snap-style attention from those queries to the allowed key-range.
        - Smooth with avg_pool1d and average across KV groups (as in SnapKV).
        - Write scores into the corresponding key positions.
        - Protect the observation window itself by assigning its max score.

      Finally, we merge the two passes into a single (B, kv_heads, seq_len) score tensor via assignment (ranges are disjoint)
    """

    defense_span: tuple[int, int] | None = None  # [start, end)
    sys_instr_span: tuple[int, int] | None = None  # [start, end)

    def _scores_from_attn_window(
        self,
        attentions: torch.Tensor,  # (B, num_heads, q_len, q_len)
        num_kv_heads: int,
        num_kv_groups: int,
        query_start: int,
        query_end: int,
        key_start: int,
        key_end: int,
        kernel_size: int,
    ) -> torch.Tensor:
        """
        Slice provided attentions for queries [query_start:query_end) over keys [key_start:key_end),
        then SnapKV-postprocess to (B, kv_heads, key_len).
        """
        key_len = max(0, key_end - key_start)
        if key_len <= 0 or query_end <= query_start:
            # return correctly-shaped zeros so callers can no-op merge
            bsz = attentions.size(0)
            return attentions.new_zeros((bsz, num_kv_heads, max(0, key_len)))

        # (B, H, Qwin, Kwin)
        attn_window = attentions[..., query_start:query_end, key_start:key_end]

        # average over query positions (B, H, Kwin)
        scores = attn_window.mean(dim=-2)

        # smooth like SnapKV
        scores = F.avg_pool1d(
            scores, kernel_size=kernel_size, padding=kernel_size // 2, stride=1
        )

        # average over KV groups: H = kv_heads * groups
        bsz = scores.size(0)
        scores = scores.view(bsz, num_kv_heads, num_kv_groups, key_len).mean(
            2
        )  # (B, kv_heads, Kwin)
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
        bsz, num_kv_heads, q_len, _ = keys.shape
        assert self.defense_span is not None and self.sys_instr_span is not None
        defense_span_start, defense_span_end = self.defense_span
        sys_instr_span_start, sys_instr_span_end = self.sys_instr_span
        for s, e, name in [
            (defense_span_start, defense_span_end, "defense_span"),
            (sys_instr_span_start, sys_instr_span_end, "sys_instr_span"),
        ]:
            assert 0 <= s <= e <= q_len, f"Invalid {name} {s, e} for q_len={q_len}"

        if attentions is None:
            raise ValueError(
                "Attentions are required for SnapKVFairEvictionPress. Set output_attentions=True and attn_implementation='eager' to use this hook"
            )

        # adjacency + order
        if defense_span_end == sys_instr_span_start:
            defense_first = True
        elif sys_instr_span_end == defense_span_start:
            defense_first = False
        else:
            raise AssertionError("Spans must be adjacent and non-overlapping")

        if defense_first:
            earlier_span_start, earlier_span_end = defense_span_start, defense_span_end
            later_span_start, later_span_end = sys_instr_span_start, sys_instr_span_end
        else:
            earlier_span_start, earlier_span_end = (
                sys_instr_span_start,
                sys_instr_span_end,
            )
            later_span_start, later_span_end = defense_span_start, defense_span_end

        earlier_span_start = 0
        later_span_end = q_len
        earlier_span_len = max(0, earlier_span_end - earlier_span_start)
        later_span_len = max(0, later_span_end - later_span_start)

        # split window_size into two halves, clamped by span lengths
        num_heads = module.config.num_attention_heads  # type: ignore
        num_kv_groups: int = num_heads // num_kv_heads  # type: ignore

        we = min(self.window_size // 2, earlier_span_len)
        wl = min(self.window_size - we, later_span_len)

        # earlier window = last we tokens of earlier span
        earlier_window_end = earlier_span_end
        earlier_window_start = max(earlier_span_end - we, earlier_span_start)
        # keys voted on: earlier span before the window
        earlier_keys_start = earlier_span_start
        earlier_keys_end = max(earlier_window_start, earlier_keys_start)

        # later window = last wl tokens of later span
        later_window_end = later_span_end
        later_window_start = max(later_span_end - wl, later_span_start)
        # keys voted on: later span before the window
        later_keys_start = later_span_start
        later_keys_end = max(later_window_start, later_keys_start)

        # init result
        scores = keys[..., 0].new_zeros((bsz, num_kv_heads, q_len))

        # earlier span pass
        scores_earlier = self._scores_from_attn_window(
            attentions=attentions,
            num_kv_heads=num_kv_heads,
            num_kv_groups=num_kv_groups,
            query_start=earlier_window_start,
            query_end=earlier_window_end,
            key_start=earlier_keys_start,
            key_end=earlier_keys_end,
            kernel_size=self.kernel_size,
        )
        scores[:, :, earlier_keys_start:earlier_keys_end] = scores_earlier

        # protect earlier window by its own max
        if earlier_window_end > earlier_window_start:
            if scores_earlier.numel() > 0:
                earlier_keep = torch.amax(
                    scores_earlier, dim=-1, keepdim=True
                )  # (B, kv_heads, 1)
            else:
                earlier_keep = torch.ones(
                    (bsz, num_kv_heads, 1), device=scores.device, dtype=scores.dtype
                )
            scores[:, :, earlier_window_start:earlier_window_end] = earlier_keep.expand(
                -1, -1, earlier_window_end - earlier_window_start
            )

        # later span pass
        scores_later = self._scores_from_attn_window(
            attentions=attentions,
            num_kv_heads=num_kv_heads,
            num_kv_groups=num_kv_groups,
            query_start=later_window_start,
            query_end=later_window_end,
            key_start=later_keys_start,
            key_end=later_keys_end,
            kernel_size=self.kernel_size,
        )
        scores[:, :, later_keys_start:later_keys_end] = scores_later

        # protect later window by its own max
        if later_window_end > later_window_start:
            if scores_later.numel() > 0:
                later_keep = torch.amax(scores_later, dim=-1, keepdim=True)
            else:
                later_keep = torch.ones(
                    (bsz, num_kv_heads, 1), device=scores.device, dtype=scores.dtype
                )
            scores[:, :, later_window_start:later_window_end] = later_keep.expand(
                -1, -1, later_window_end - later_window_start
            )

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
