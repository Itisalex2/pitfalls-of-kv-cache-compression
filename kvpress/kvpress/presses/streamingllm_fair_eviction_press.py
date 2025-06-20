# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import torch
from torch import nn

from kvpress.presses.streaming_llm_press import StreamingLLMPress


@dataclass
class StreamingLLMFairEvictionPress(StreamingLLMPress):
    """
    StreamingLLM with special handling for system instruction and defense spans (split method).
    We treat the defense span and the system-instruction span as two adjacent windows, and
    compress/rank them separately in a StreamingLLM-like fashion, then merge.

    Sinks:
    - ONLY a prefix sink: first n_sink tokens at index 0 are set to score 1.0.

    Explanation of scores:
    - Sink tokens (first n_sink, owned by the earlier span): score 1.0
    - Defense span (expanded to its owned side): linearly increasing scores from 0.1 to 0.9
    - System instruction span (expanded to its owned side): linearly increasing scores from 0.1 to 0.9

    Assumptions:
    - Only two spans are used: defense and system-instruction.
    - Spans are adjacent (touching) and non-overlapping, in either order.
    """

    defense_span: tuple[int, int] | None = None  # [start, end)
    sys_instr_span: tuple[int, int] | None = None  # [start, end)

    def _apply_ramp(self, target_scores: torch.Tensor, start_idx: int, end_idx: int):
        """
        Build per-span ramps over their expanded ownership regions.
        """

        length = max(0, end_idx - start_idx)
        if length <= 0:
            raise ValueError(
                f"Invalid ramp length {length} for indices {start_idx}, {end_idx}"
            )
        ramp = torch.linspace(0.1, 0.9, steps=length, device=target_scores.device)
        target_scores[:, :, start_idx:end_idx] = ramp.view(1, 1, length)

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        q_len = hidden_states.shape[1]
        assert self.defense_span is not None, "defense_span must be set"
        assert self.sys_instr_span is not None, "sys_instr_span must be set"
        assert q_len > self.n_sink, (
            f"Input should contain more tokens than n_sink={self.n_sink}"
        )

        scores = torch.zeros_like(keys[..., 0])  # (batch, heads, q_len)

        # Unpack spans and validate bounds
        defense_span_start, defense_span_end = self.defense_span
        sys_instr_span_start, sys_instr_span_end = self.sys_instr_span
        for s, e, name in [
            (defense_span_start, defense_span_end, "defense_span"),
            (sys_instr_span_start, sys_instr_span_end, "sys_instr_span"),
        ]:
            assert 0 <= s <= e <= q_len, f"Invalid {name} {s, e} for q_len={q_len}"

        # Enforce adjacency & determine order (no overlap, exactly touching)
        # Accept either defense first or system-instruction first.
        if defense_span_end == sys_instr_span_start:
            defense_first = True
        elif sys_instr_span_end == defense_span_start:
            defense_first = False
        else:
            raise AssertionError(
                f"Spans must be adjacent and non-overlapping. "
                f"Got defense={self.defense_span}, sys_instr={self.sys_instr_span}"
            )

        # Expanded ownership:
        # - Earlier span owns [0, earlier_end)
        # - Later span owns [later_start, q_len)
        if defense_first:
            _, earlier_span_end = defense_span_start, defense_span_end
            later_span_start, _ = sys_instr_span_start, sys_instr_span_end
        else:
            _, earlier_span_end = (
                sys_instr_span_start,
                sys_instr_span_end,
            )
            later_span_start, _ = defense_span_start, defense_span_end

        # Earlier expanded region: [0, earlier_span_end)
        # Later expanded region: [later_span_start, q_len)
        # (We keep contiguous ramps; sinks will be set to 1.0 after and override ramp values.)
        # Apply ramps to expanded regions
        # Earlier region
        self._apply_ramp(scores, self.n_sink, earlier_span_end)
        # Later region
        self._apply_ramp(scores, later_span_start, q_len)

        # Now set sinks to 1.0, overriding ramps where they overlap.
        prefix_end = min(self.n_sink, q_len)
        scores[:, :, :prefix_end] = 1.0

        return scores
