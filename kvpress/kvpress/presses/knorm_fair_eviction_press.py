# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.knorm_press import KnormPress


@dataclass
class KnormFairEvictionPress(KnormPress):
    """
    Knorm with fair two-span eviction.
    """

    defense_span: tuple[int, int] | None = None  # [start, end)
    sys_instr_span: tuple[int, int] | None = None  # [start, end)

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
