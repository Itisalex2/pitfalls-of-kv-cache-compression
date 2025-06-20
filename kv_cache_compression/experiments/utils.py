import datetime
import json
from pathlib import Path

import numpy as np
import torch
from kvpress import (
    PyramidKVPress,
    ScorerPress,
    SnapKVPress,
    StreamingLLMPress,
    ObservedAttentionPress,
    TOVAPress,
    TOVAFairEvictionPress,
    KnormPress,
    KnormFairEvictionPress,
    ObservedAttentionFairEvictionPress,
    SnapKVFairEvictionPress,
    StreamingLLMFairEvictionPress,
)
import os
from transformers.models.auto.tokenization_auto import AutoTokenizer


def get_attention_implementation_from_strategy(strategy: str) -> str:
    """
    Returns the attention implementation based on the press strategy.
    """
    eager_attention_strategies = [
        "observed_attention",
        "observed_attention_fair_eviction",
        "snap_fair_eviction",
        "tova_fair_eviction",
    ]
    if strategy in eager_attention_strategies:
        return "eager"
    else:
        return "flash_attention_2"


def get_output_attentions_from_strategy(strategy: str) -> bool:
    """
    Returns whether the model should output attentions based on the press strategy.
    """
    output_attentions_strategies = [
        "observed_attention",
        "observed_attention_fair_eviction",
        "snap_fair_eviction",
        "tova_fair_eviction",
    ]
    if strategy in output_attentions_strategies:
        return True
    else:
        return False


def default_json_serializer(o) -> int | float | list:
    """Custom JSON serializer for objects not serializable by default json code."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.cpu().detach().numpy().tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def analyze_and_save_results(
    model_name: str,
    dataset_name: str,
    eval_name: str,
    prompt_results_list: list,
    all_scores_flat: list,
    base_output_dir: Path,
    logger,
) -> dict:
    """
    Analyzes results from a batch of prompts and saves an aggregated JSON file.
    """
    overall_stats = {}
    if all_scores_flat:
        overall_mean = np.mean(all_scores_flat)
        overall_std = np.std(all_scores_flat)
        overall_sem = overall_std / np.sqrt(len(all_scores_flat))
        overall_min = np.min(all_scores_flat)
        overall_max = np.max(all_scores_flat)
        overall_stats = {
            "mean": overall_mean,
            "std": overall_std,
            "sem": overall_sem,
            "min": overall_min,
            "max": overall_max,
        }
    else:
        overall_stats = {
            "mean": 0,
            "std": 0,
            "sem": 0,
            "min": 0,
            "max": 0,
        }

    prompt_results_list.sort(key=lambda x: x["evaluation"]["mean_score"], reverse=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_name = f"aggregated_{dataset_name}_{eval_name}_{timestamp}.json"
    output_file_path = base_output_dir / output_file_name

    analysis_data = {
        "model": model_name,
        "dataset": dataset_name,
        "eval_name": eval_name,
        "overall_stats": overall_stats,
        "prompt_results": prompt_results_list,
        "generated_at": timestamp,
    }

    with open(output_file_path, "w") as f:
        json.dump(analysis_data, f, indent=2, default=default_json_serializer)

    logger.info(f"Aggregated analysis for {eval_name} saved to {output_file_path}")
    logger.info(
        f"Overall Score for {eval_name}: {overall_stats.get('mean', 0):.4f} ± {overall_stats.get('sem', 0):.4f}"
    )

    return overall_stats  # Return for potential use (e.g., plotting)


def press_resolver(strategy: str) -> ScorerPress:
    """
    Resolves the press strategy to the appropriate class instance.
    By default, compression = 0.0. It should be modified after instantiation
    For pyramid and snap, the window size is set to 16.
    """
    if strategy == "pyramid":
        return PyramidKVPress(window_size=16)
    elif strategy == "streamingllm":
        return StreamingLLMPress()
    elif strategy == "snap":
        return SnapKVPress(window_size=16)
    elif strategy == "observed_attention":
        return ObservedAttentionPress()
    elif strategy == "tova":
        return TOVAPress()
    elif strategy == "tova_fair_eviction":
        return TOVAFairEvictionPress()
    elif strategy == "knorm":
        return KnormPress()
    elif strategy == "knorm_fair_eviction":
        return KnormFairEvictionPress()
    elif strategy == "observed_attention_fair_eviction":
        return ObservedAttentionFairEvictionPress()
    elif strategy == "snap_fair_eviction":
        return SnapKVFairEvictionPress(window_size=16)
    elif strategy == "streamingllm_fair_eviction":
        return StreamingLLMFairEvictionPress()
    else:
        raise NotImplementedError(f"Press strategy not implemented: {strategy}")


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_jsonl(path: Path | str, num_items: int) -> list[dict]:
    """Read a jsonl file and return a list of dictionaries."""
    assert num_items != 0, "num_items must be non-zero"
    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if num_items > 0 and len(items) >= num_items:
                break
    return items


def write_jsonl(path: Path | str, rows: list[dict], append: bool = False) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if append:
        mode = "a"
    else:
        mode = "w"
    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_system_span(
    tokenizer: AutoTokenizer, full_ids: list[int], system_prompt: str
) -> tuple[int, int]:
    sys_ids = tokenizer.encode(  # type: ignore
        system_prompt, add_special_tokens=False
    )
    # Start and end system prompt tokens may join with adjacent tokens. We slice them off for matching.
    sys_ids_sliced = sys_ids[1:-1]
    span = find_subsequence(full_ids, sys_ids_sliced)

    # Expand span to include the sliced-off tokens
    span = (span[0] - 1, span[1] + 1) if span is not None else None
    if span is None:
        raise ValueError(
            "System prompt tokens not found as a subsequence in prefill ids."
        )
    return span


def find_subsequence(haystack: list[int], needle: list[int]) -> tuple[int, int] | None:
    if not needle or len(needle) > len(haystack):
        return None

    first = needle[0]
    Lh, Ln = len(haystack), len(needle)
    i = 0
    while i <= Lh - Ln:
        # quick skip to next candidate
        while i <= Lh - Ln and haystack[i] != first:
            i += 1
        if i > Lh - Ln:
            break
        if haystack[i : i + Ln] == needle:
            return i, i + Ln
        i += 1
    return None


def keep_rates_from_kept_indices(
    kept_indices_by_layer: dict[int, torch.Tensor],
    seq_len: int,
) -> tuple[dict[int, list[float]], list[float]]:
    """
    kept_indices_by_layer[layer]: Tensor[B, H, n_kept]. Assumes B==1.
    Returns:
    per_layer_rates: {layer -> [seq_len floats]}, each float in [0,1]
    overall_rate:    [seq_len floats], mean across layers
    """
    per_layer_rates: dict[int, list[float]] = {}
    layer_vectors: list[torch.Tensor] = []

    for layer_idx in sorted(kept_indices_by_layer.keys()):
        kept = kept_indices_by_layer[layer_idx]
        kept = kept.detach().cpu()
        assert kept.dim() == 3 and kept.shape[0] == 1, "Expected shape (B=1, H, n_kept)"
        H, _ = kept.shape[1], kept.shape[2]

        # mask[h, t] = True if head h kept token position t
        mask = torch.zeros((H, seq_len), dtype=torch.bool)
        mask.scatter_(1, kept[0], True)  # kept[0]: (H, n_kept)

        keep_rate = mask.float().mean(dim=0)  # (seq_len,)
        per_layer_rates[layer_idx] = keep_rate.tolist()
        layer_vectors.append(keep_rate.unsqueeze(0))

    overall_rate = (
        torch.cat(layer_vectors, dim=0).mean(dim=0).tolist()
        if layer_vectors
        else [0.0] * seq_len
    )
    return per_layer_rates, overall_rate


def segment_keep_pcts(
    overall_keep_rate: list[float],
    system_span: tuple[int, int],
    seq_len: int,
) -> tuple[float, float]:
    """
    Returns (system_keep_pct, defense_keep_pct), both in [0,1].
    system_span is [start, end) in prefill token space.
    """
    ov = torch.tensor(overall_keep_rate)
    sys_start, sys_end = system_span
    assert 0 <= sys_start <= sys_end <= seq_len, "Bad system span vs seq_len"

    system_len = sys_end - sys_start
    if system_len > 0:
        system_keep = float(ov[sys_start:sys_end].mean().item())
    else:
        system_keep = 0.0

    # defense = everything not system
    defense_mask = torch.ones(seq_len, dtype=torch.bool)
    if system_len > 0:
        defense_mask[sys_start:sys_end] = False
    defense_keep = float(ov[defense_mask].mean().item()) if defense_mask.any() else 0.0

    return system_keep, defense_keep
