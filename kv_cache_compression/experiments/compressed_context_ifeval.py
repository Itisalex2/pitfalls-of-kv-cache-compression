import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from jsonargparse import CLI
from kvpress import ScorerPress
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from ..benchmarks.ifeval import evaluate as ifeval
from .compressed_context_evaluation import (
    CompressedContextConfig,
    generate_from_compressed_context,
    get_logits_entropy,
    get_context_cache,
)
from .utils import (
    get_attention_implementation_from_strategy,
    get_output_attentions_from_strategy,
    default_json_serializer,
    press_resolver,
    read_jsonl,
    set_seeds,
    write_jsonl,
    get_system_span,
    keep_rates_from_kept_indices,
    segment_keep_pcts,
)


@dataclass
class CompressedContextIFEval(CompressedContextConfig):
    """
    Configuration for compressed context evaluation with IFEval dataset.
    IFEval is modified to include system prompt messages.
    Ref: https://github.com/normster/SystemCheck/tree/main/evals/ifeval
    """

    outputs_base: Path = (
        Path(__file__).resolve().parent / "compressed_context_ifeval_outputs"
    )
    sys_ifeval_path: Path = (
        Path(__file__).resolve().parent.parent
        / "benchmarks/ifeval"
        / "inputs"
        / "sys_ifeval.jsonl"
    )
    num_prompts: int = -1  # -1 means all prompts in the dataset
    sample_responses: bool = (
        False  # Whether to sample responses or use greedy decoding.
    )
    generate_entropy: bool = True
    skip_ids: list[str] | None = None
    max_new_tokens: int = 1280
    system_prompt_prepend: str | None = (
        "You are a helpful assistant. If you receive a valid user request, please respond while adhering to the following guidelines:\n\n"  # Optional string to prepend to all system prompts.
    )
    defense_template_key: str | None = None
    defense_template_path: Path = (
        Path(__file__).resolve().parent.parent
        / "benchmarks"
        / "raccoon"
        / "defenses"
        / "defense_template.json"
    )
    force_keep_global: list[int] | None = (
        None  # List of prefill token positions to always keep
    )
    analyze_kept_tokens: bool = True
    defense_span: tuple[int, int] | None = (
        None  # (start inclusive, end exclusive) indices of defense in the prompt. Used only with SystemPromptPress.
    )
    use_automated_spans: bool = False  # For fair eviction methods only. If True, use automated detection of defense and system instruction spans.


def if_eval_read_prompt_list_override(
    p: Path, skip_ids: list[str] | None
) -> list[ifeval.InputExample]:
    items = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            if skip_ids:
                if set(ex["instruction_id_list"]).intersection(skip_ids):
                    continue
            messages = ex.get("messages")
            if not messages:
                raise ValueError("Each example must have 'messages' field.")
            items.append(
                ifeval.InputExample(
                    key=ex["key"],
                    instruction_id_list=ex["instruction_id_list"],
                    messages=messages,
                    kwargs=ex["kwargs"],
                )
            )
    return items


def get_sys_ifeval_responses_at_ratio(
    ratio: float,
    sys_ifeval_items: list[dict],
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    press: ScorerPress,
    config: CompressedContextIFEval,
) -> list[dict]:
    responses: list[dict] = []
    device = next(model.parameters()).device
    logger = logging.getLogger(__name__)

    for item in tqdm(
        sys_ifeval_items, desc=f"Generating responses at ratio {ratio:.2f}"
    ):
        try:
            assert item["messages"][0]["role"] == "system", (
                "First message must be system prompt"
            )
            system_prompt_content_with_template = [item["messages"][0]]
            system_prompt_content_ids: torch.Tensor = tokenizer.apply_chat_template(  # type:ignore
                system_prompt_content_with_template, return_tensors="pt"
            ).to(device)
            system_prompt_content_len = system_prompt_content_ids.shape[1]

            if config.analyze_kept_tokens:
                system_prompt_span = get_system_span(
                    tokenizer,
                    system_prompt_content_ids[0].tolist(),
                    item["original_messages"][0]["content"],
                )
                press.clear_analysis()
            else:
                system_prompt_span = None

            if config.use_automated_spans:
                # Automatically detect defense and system instruction spans for fair eviction methods
                defense_template_without_system_prompt = item[
                    "defense_template"
                ].replace("$system_prompt", "")

                defense_span = get_system_span(
                    tokenizer,
                    system_prompt_content_ids[0].tolist(),
                    defense_template_without_system_prompt,
                )
                sys_instr_span = get_system_span(
                    tokenizer,
                    system_prompt_content_ids[0].tolist(),
                    item["original_messages"][0]["content"],
                )
                if (
                    defense_span[1] != sys_instr_span[0]
                    and defense_span[0] != sys_instr_span[1]
                ):
                    if sys_instr_span[1] == defense_span[0] + 1:
                        logger.info(
                            f"Adjusting faulty system instruction span to be contiguous with defense span: {defense_span} and {sys_instr_span}. Key: {item['key']}."
                        )
                        sys_instr_span = (sys_instr_span[0], defense_span[0])
                        logger.info(
                            f"New system instruction span: {sys_instr_span}. Key: {item['key']}."
                        )
                    else:
                        logger.warning(
                            f"Automated spans for defense and system prompt are overlapping or non-contiguous: {defense_span} and {sys_instr_span}. Key: {item['key']}. This may affect eviction methods that rely on these spans."
                        )
                assert hasattr(press, "defense_span"), (
                    "Press must have defense_span attribute if use_automated_spans is True"
                )
                assert hasattr(press, "sys_instr_span"), (
                    "Press must have sys_instr_span attribute if use_automated_spans is True"
                )
                press.defense_span = defense_span  # type: ignore
                press.sys_instr_span = sys_instr_span  # type: ignore

            context_cache = get_context_cache(model, system_prompt_content_ids, press)
            context_cache_len = context_cache[0][0].shape[2]  # type: ignore

            if config.analyze_kept_tokens:
                assert isinstance(system_prompt_span, tuple)
                per_layer_keep_rate, overall_keep_rate = keep_rates_from_kept_indices(
                    press.kept_indices_by_layer,
                    system_prompt_content_len,
                )

                # Compression ratio 0.0
                if not press.kept_indices_by_layer:
                    overall_keep_rate = [1.0] * system_prompt_content_len

                system_keep_percentage, defense_keep_percentage = segment_keep_pcts(
                    overall_keep_rate,
                    system_prompt_span,
                    system_prompt_content_len,
                )
            else:
                per_layer_keep_rate = {}
                overall_keep_rate = []
                system_keep_percentage = -1.0
                defense_keep_percentage = -1.0

            messages = item["messages"]
            messages_ids = tokenizer.apply_chat_template(  # type: ignore
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(device)

            generated_ids, logits, logits_non_special_tokens_mask = (
                generate_from_compressed_context(
                    model,
                    tokenizer,
                    messages_ids,
                    system_prompt_content_len,
                    context_cache,
                    do_sample=config.sample_responses,
                    max_new_tokens=config.max_new_tokens,
                )
            )

            if config.generate_entropy:
                entropy = get_logits_entropy(logits, logits_non_special_tokens_mask)
            else:
                entropy = None

            response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)  # type: ignore

            system_prompt_content_str_tokens = tokenizer.convert_ids_to_tokens(  # type: ignore
                system_prompt_content_ids[0].tolist(),
                skip_special_tokens=False,
            )
            system_prompt_ids_and_str_tokens = list(
                zip(
                    range(system_prompt_content_len),
                    system_prompt_content_ids[0].tolist(),
                    system_prompt_content_str_tokens,
                )
            )  # type: ignore

            response_entry = {
                "key": item["key"],
                "messages": item["messages"]
                + [{"role": "assistant", "content": response}],
                "response_entropy": entropy,
                "system_prompt_token_span": system_prompt_span,
                "system_prompt_ids_and_str_tokens": system_prompt_ids_and_str_tokens,
                "system_prompt_cache_length": context_cache_len,
                "keep_rate_per_layer": per_layer_keep_rate,
                "keep_rate_overall": overall_keep_rate,
                "system_keep_pct": system_keep_percentage,
                "defense_keep_pct": defense_keep_percentage,
            }

            responses.append(response_entry)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(
                f"Error generating response for item {item.get('key', '<no-key>')}: {e}"
            )
            logger.error(f"Skipping item {item.get('key', '<no-key>')}.")

    return responses


def get_ifeval_results(
    input_path: Path,
    response_path: Path,
    skip_ids: list[str] | None = None,
    tokenizer: AutoTokenizer | None = None,
) -> dict:
    prompt_to_response = ifeval.read_prompt_to_response_dict(response_path)
    inputs = if_eval_read_prompt_list_override(input_path, skip_ids=skip_ids)

    # Filter out any inputs for which we don't have a response
    inputs_eval = [inp for inp in inputs if inp.key in prompt_to_response]
    missing = len(inputs) - len(inputs_eval)

    outputs_by_version: dict[str, list[ifeval.OutputExample]] = {}
    results = defaultdict(list)
    for func, version in [
        (ifeval.test_instruction_following_strict, "strict"),
        (ifeval.test_instruction_following_loose, "loose"),
    ]:
        outputs: list[ifeval.OutputExample] = []
        for inp in inputs_eval:
            output = func(inp, prompt_to_response)
            outputs.append(output)
            results[f"prompt_{version}"].append(output.follow_all_instructions)
            results[f"instruction_{version}"].append(output.follow_instruction_list)
        outputs_by_version[version] = outputs

    scores = ifeval.compute_scores(results)
    n_prompts = len(inputs_eval)
    n_instructions = sum(len(x) for x in results["instruction_strict"])

    # Per-instruction breakdown
    # Map instruction_id -> counts
    per_instruction = {}  # id -> {"n": int, "strict_correct": int, "loose_correct": int}
    for idx, inp in enumerate(inputs_eval):
        strict_out = outputs_by_version["strict"][idx]
        loose_out = outputs_by_version["loose"][idx]
        for j, inst_id in enumerate(inp.instruction_id_list):
            rec = per_instruction.setdefault(
                inst_id, {"n": 0, "strict_correct": 0, "loose_correct": 0}
            )
            rec["n"] += 1
            if strict_out.follow_instruction_list[j]:
                rec["strict_correct"] += 1
            if loose_out.follow_instruction_list[j]:
                rec["loose_correct"] += 1

    # Convert to accuracies
    by_instruction = {}
    for inst_id, rec in per_instruction.items():
        n = max(rec["n"], 1)
        by_instruction[inst_id] = {
            "n": rec["n"],
            "strict": rec["strict_correct"] / n,
            "loose": rec["loose_correct"] / n,
        }

    # Per-family breakdown e.g. keywords, language
    per_family = defaultdict(lambda: {"n": 0, "strict_correct": 0, "loose_correct": 0})
    for inst_id, rec in per_instruction.items():
        fam = inst_id.split(":")[0] if ":" in inst_id else inst_id
        agg = per_family[fam]
        agg["n"] += rec["n"]
        agg["strict_correct"] += rec["strict_correct"]
        agg["loose_correct"] += rec["loose_correct"]

    by_family = {
        fam: {
            "n": rec["n"],
            "strict": (rec["strict_correct"] / rec["n"]) if rec["n"] else 0.0,
            "loose": (rec["loose_correct"] / rec["n"]) if rec["n"] else 0.0,
        }
        for fam, rec in per_family.items()
    }

    # SEM calculations
    def _fam_of(inst_id: str) -> str:
        return inst_id.split(":")[0] if ":" in inst_id else inst_id

    # Collect raw 0/1 correctness for each family and overall
    fam_lists: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"strict": [], "loose": []}
    )
    overall_strict: list[int] = []
    overall_loose: list[int] = []

    for idx, inp in enumerate(inputs_eval):
        strict_out = outputs_by_version["strict"][idx]
        loose_out = outputs_by_version["loose"][idx]
        fams = [_fam_of(iid) for iid in inp.instruction_id_list]
        s_list = strict_out.follow_instruction_list
        l_list = loose_out.follow_instruction_list
        for fam, s_ok, l_ok in zip(fams, s_list, l_list):
            fam_lists[fam]["strict"].append(1 if s_ok else 0)
            fam_lists[fam]["loose"].append(1 if l_ok else 0)
            overall_strict.append(1 if s_ok else 0)
            overall_loose.append(1 if l_ok else 0)

    def _sem(vals: list[int]) -> float:
        if not vals:
            return float("nan")
        if len(vals) == 1:
            return 0.0
        arr = np.array(vals, dtype=float)
        return float(arr.std(ddof=1) / np.sqrt(len(arr)))

    # Attach SEM to by_family
    for fam, rec in by_family.items():
        strict_vals = fam_lists.get(fam, {}).get("strict", [])
        loose_vals = fam_lists.get(fam, {}).get("loose", [])
        rec["strict_err"] = _sem(strict_vals)
        rec["loose_err"] = _sem(loose_vals)

    # Overall SEM (instruction-level), add to scores
    scores["overall_err_strict"] = _sem(overall_strict)
    scores["overall_err_loose"] = _sem(overall_loose)

    # system-prompt-length analysis
    # compute system token length per prompt (using the same chat template)
    sys_token_lens: list[int] = []
    if tokenizer is not None:
        for inp in inputs_eval:
            sys_only = [inp.messages[0]]  # first message is "system"
            ids = tokenizer.apply_chat_template(sys_only, return_tensors="pt")  # type: ignore
            sys_token_lens.append(int(ids.shape[1]))
    else:
        # fallback: character length if no tokenizer provided
        sys_token_lens = [len(inp.messages[0]["content"]) for inp in inputs_eval]

    L = np.array(sys_token_lens, dtype=int)
    # make up to 4 quantile bins; dedupe edges to avoid empty/degenerate bins
    q = np.unique(np.quantile(L, np.linspace(0, 1, 5), method="nearest")).astype(int)
    if q.size < 2:
        q = np.array([L.min(), L.max()], dtype=int)

    # bin accumulators
    bin_stats = [
        {
            "range": [int(q[i]), int(q[i + 1])],
            "n_prompts": 0,
            "n_instructions": 0,
            "prompt_strict_correct": 0,
            "prompt_loose_correct": 0,
            "instr_strict_correct": 0,
            "instr_loose_correct": 0,
        }
        for i in range(len(q) - 1)
    ]

    def bin_index(x: int) -> int:
        # right-closed top bin; place x in the last bin if equals max
        for i in range(len(q) - 1):
            lo, hi = q[i], q[i + 1]
            if (x >= lo) and (x < hi or (i == len(q) - 2 and x <= hi)):
                return i
        return len(q) - 2  # safety

    # aggregate per prompt
    for idx, inp in enumerate(inputs_eval):
        b = bin_index(L[idx])
        strict_out = outputs_by_version["strict"][idx]
        loose_out = outputs_by_version["loose"][idx]

        bin_stats[b]["n_prompts"] += 1
        bin_stats[b]["prompt_strict_correct"] += int(strict_out.follow_all_instructions)
        bin_stats[b]["prompt_loose_correct"] += int(loose_out.follow_all_instructions)

        # instruction-level: sum booleans and counts
        bin_stats[b]["n_instructions"] += len(strict_out.follow_instruction_list)
        bin_stats[b]["instr_strict_correct"] += int(
            np.sum(strict_out.follow_instruction_list)
        )
        bin_stats[b]["instr_loose_correct"] += int(
            np.sum(loose_out.follow_instruction_list)
        )

    # turn counts into rates
    by_syslen_bins = {"edges": [int(x) for x in q.tolist()], "bins": []}
    for rec in bin_stats:
        n_p = rec["n_prompts"]
        n_i = rec["n_instructions"]
        by_syslen_bins["bins"].append(
            {
                "range": rec["range"],  # [low_tokens, high_tokens]
                "n_prompts": n_p,
                "n_instructions": n_i,
                "prompt_strict": (rec["prompt_strict_correct"] / n_p)
                if n_p
                else float("nan"),
                "prompt_loose": (rec["prompt_loose_correct"] / n_p)
                if n_p
                else float("nan"),
                "instruction_strict": (rec["instr_strict_correct"] / n_i)
                if n_i
                else float("nan"),
                "instruction_loose": (rec["instr_loose_correct"] / n_i)
                if n_i
                else float("nan"),
            }
        )

    # --- Per-prompt records for co-occurrence analysis ---
    per_prompt = []
    for idx, inp in enumerate(inputs_eval):
        strict_out = outputs_by_version["strict"][idx]
        loose_out = outputs_by_version["loose"][idx]
        fams = [
            iid.split(":")[0] if ":" in iid else iid for iid in inp.instruction_id_list
        ]
        per_prompt.append(
            {
                "key": inp.key,
                "instruction_ids": inp.instruction_id_list,
                "families": fams,  # parallel to instruction_ids
                "strict_list": strict_out.follow_instruction_list,  # bool per instruction
                "loose_list": loose_out.follow_instruction_list,  # bool per instruction
            }
        )

    # --- Single- vs Multi-instruction group aggregates ---
    # Identify prompt indices by number of instructions
    single_idx = [
        i for i, inp in enumerate(inputs_eval) if len(inp.instruction_id_list) == 1
    ]
    multi_idx = [
        i for i, inp in enumerate(inputs_eval) if len(inp.instruction_id_list) >= 2
    ]

    def _group_agg(indices: list[int]) -> dict:
        """Aggregate prompt- and instruction-level metrics, plus per-family, over a subset of prompts."""
        if not indices:
            return {
                "prompt_strict": float("nan"),
                "prompt_loose": float("nan"),
                "instruction_strict": float("nan"),
                "instruction_loose": float("nan"),
                "prompt_err_strict": float("nan"),
                "prompt_err_loose": float("nan"),
                "instruction_err_strict": float("nan"),
                "instruction_err_loose": float("nan"),
                "by_family": {},
                "counts": {"prompts": 0, "instructions": 0},
            }

        # Collect raw 0/1 lists for prompt-level accuracy and instruction-level accuracy
        prompt_strict_vals, prompt_loose_vals = [], []
        instr_strict_vals, instr_loose_vals = [], []

        # Per-family raw lists
        fam_lists = defaultdict(lambda: {"strict": [], "loose": []})

        for idx in indices:
            strict_out = outputs_by_version["strict"][idx]
            loose_out = outputs_by_version["loose"][idx]
            inp = inputs_eval[idx]
            fams = [_fam_of(iid) for iid in inp.instruction_id_list]

            # prompt-level booleans
            prompt_strict_vals.append(1 if strict_out.follow_all_instructions else 0)
            prompt_loose_vals.append(1 if loose_out.follow_all_instructions else 0)

            # instruction-level booleans
            s_list = [1 if b else 0 for b in strict_out.follow_instruction_list]
            l_list = [1 if b else 0 for b in loose_out.follow_instruction_list]
            instr_strict_vals.extend(s_list)
            instr_loose_vals.extend(l_list)

            # per-family split
            for fam, s_ok, l_ok in zip(
                fams,
                strict_out.follow_instruction_list,
                loose_out.follow_instruction_list,
            ):
                fam_lists[fam]["strict"].append(1 if s_ok else 0)
                fam_lists[fam]["loose"].append(1 if l_ok else 0)

        # Build by_family with means + SEM
        by_family_group = {}
        for fam, rec in fam_lists.items():
            s_vals = rec["strict"]
            l_vals = rec["loose"]
            by_family_group[fam] = {
                "n": len(s_vals),  # (# instructions of this family within group)
                "strict": float(np.mean(s_vals)) if s_vals else float("nan"),
                "loose": float(np.mean(l_vals)) if l_vals else float("nan"),
                "strict_err": _sem(s_vals) if s_vals else float("nan"),
                "loose_err": _sem(l_vals) if l_vals else float("nan"),
            }

        return {
            "prompt_strict": float(np.mean(prompt_strict_vals))
            if prompt_strict_vals
            else float("nan"),
            "prompt_loose": float(np.mean(prompt_loose_vals))
            if prompt_loose_vals
            else float("nan"),
            "instruction_strict": float(np.mean(instr_strict_vals))
            if instr_strict_vals
            else float("nan"),
            "instruction_loose": float(np.mean(instr_loose_vals))
            if instr_loose_vals
            else float("nan"),
            "prompt_err_strict": _sem(prompt_strict_vals)
            if prompt_strict_vals
            else float("nan"),
            "prompt_err_loose": _sem(prompt_loose_vals)
            if prompt_loose_vals
            else float("nan"),
            "instruction_err_strict": _sem(instr_strict_vals)
            if instr_strict_vals
            else float("nan"),
            "instruction_err_loose": _sem(instr_loose_vals)
            if instr_loose_vals
            else float("nan"),
            "by_family": by_family_group,
            "counts": {
                "prompts": len(indices),
                "instructions": len(instr_strict_vals),
            },
        }

    single_group = _group_agg(single_idx)
    multi_group = _group_agg(multi_idx)

    return {
        "scores": {
            "prompt_strict": scores["prompt_strict"],
            "instruction_strict": scores["instruction_strict"],
            "prompt_loose": scores["prompt_loose"],
            "instruction_loose": scores["instruction_loose"],
            "overall_err_strict": scores["overall_err_strict"],
            "overall_err_loose": scores["overall_err_loose"],
        },
        "counts": {
            "prompts_evaluated": n_prompts,
            "instructions_evaluated": n_instructions,
            "missing_responses": missing,
        },
        "by_instruction": by_instruction,
        "by_family": by_family,
        "by_syslen_bins": by_syslen_bins,
        "per_prompt": per_prompt,
        "meta": {
            "input_path": str(input_path),
            "response_path": str(response_path),
            "skip_ids": skip_ids or [],
            "bootstrap": 0,
        },
        "single_multi": {
            "single": {
                "prompt_strict": single_group["prompt_strict"],
                "instruction_strict": single_group["instruction_strict"],
                "prompt_loose": single_group["prompt_loose"],
                "instruction_loose": single_group["instruction_loose"],
                "prompt_err_strict": single_group["prompt_err_strict"],
                "prompt_err_loose": single_group["prompt_err_loose"],
                "instruction_err_strict": single_group["instruction_err_strict"],
                "instruction_err_loose": single_group["instruction_err_loose"],
                "counts": single_group["counts"],
            },
            "multi": {
                "prompt_strict": multi_group["prompt_strict"],
                "instruction_strict": multi_group["instruction_strict"],
                "prompt_loose": multi_group["prompt_loose"],
                "instruction_loose": multi_group["instruction_loose"],
                "prompt_err_strict": multi_group["prompt_err_strict"],
                "prompt_err_loose": multi_group["prompt_err_loose"],
                "instruction_err_strict": multi_group["instruction_err_strict"],
                "instruction_err_loose": multi_group["instruction_err_loose"],
                "counts": multi_group["counts"],
            },
        },
        "by_family_single": single_group["by_family"],
        "by_family_multi": multi_group["by_family"],
    }


def save_responses(path: Path, responses: list[dict]) -> None:
    write_jsonl(path, responses)


def save_ifeval_results(path: Path, results: dict) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=4, default=default_json_serializer)


def plot_results_across_ratios(results_all_ratios: dict, plot_path: Path) -> None:
    """
    results_all_ratios: {
        "ratio_0.00": { "scores": {...}, "by_family": {...}, ... },
        "ratio_0.10": { ... },
        ...
    }
    Saves:
      plots/prompt_strict_vs_ratio.png
      plots/prompt_loose_vs_ratio.png
      plots/instruction_strict_vs_ratio.png
      plots/instruction_loose_vs_ratio.png
      plots/families/family_<family>_vs_ratio.png  (strict + loose)
      plots/families_all_strict_vs_ratio.png
      plots/families_all_loose_vs_ratio.png
      plots/syslen/<metric>_by_syslen_vs_ratio.png (if by_syslen_bins present)
      plots/cooccur/<famA>_vs_<famB>_strict_vs_ratio.png
    """

    plot_path.mkdir(parents=True, exist_ok=True)

    # Sort keys by numeric ratio suffix, e.g. "ratio_0.25" -> 0.25
    ratio_keys = sorted(
        results_all_ratios.keys(), key=lambda k: float(k.split("_", 1)[1])
    )
    xs = [float(k.split("_", 1)[1]) for k in ratio_keys]
    rows = [results_all_ratios[k] for k in ratio_keys]

    # --- Overall curves ---
    overall_metrics = [
        "prompt_strict",
        "prompt_loose",
        "instruction_strict",
        "instruction_loose",
    ]
    series = {m: [] for m in overall_metrics}
    for row in rows:
        s = row.get("scores", {})
        for m in overall_metrics:
            series[m].append(s.get(m, np.nan))

    for m in overall_metrics:
        ys = np.array(series[m], dtype=float)
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Compression ratio")
        pretty = m.replace("_", " ").title()
        plt.ylabel(pretty)
        plt.title(f"{pretty} vs Compression Ratio")
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_path / f"{m}_vs_ratio.png", bbox_inches="tight")
        plt.close()

    # --- Per-family curves (strict + loose on same chart) ---
    fam_dir = plot_path / "families"
    fam_dir.mkdir(parents=True, exist_ok=True)

    # collect union of families across all ratios
    families = set()
    for row in rows:
        byfam = row.get("by_family") or {}
        families.update(byfam.keys())

    for fam in sorted(families):
        strict_vals, loose_vals = [], []
        for row in rows:
            rec = (row.get("by_family") or {}).get(fam)
            if rec is None:
                strict_vals.append(np.nan)
                loose_vals.append(np.nan)
            else:
                strict_vals.append(rec.get("strict", np.nan))
                loose_vals.append(rec.get("loose", np.nan))

        strict_vals = np.array(strict_vals, dtype=float)
        loose_vals = np.array(loose_vals, dtype=float)

        # Skip if absolutely nothing to plot
        if np.all(np.isnan(strict_vals)) and np.all(np.isnan(loose_vals)):
            continue

        plt.figure()
        plt.plot(xs, strict_vals, marker="o", label="strict")
        plt.plot(xs, loose_vals, marker="s", label="loose")
        plt.xlabel("Compression ratio")
        plt.ylabel("Accuracy")
        plt.title(f"{fam} Accuracy vs Compression Ratio")
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend()
        safe = fam.replace(":", "_").replace("/", "_")
        plt.savefig(fam_dir / f"family_{safe}_vs_ratio.png", bbox_inches="tight")
        plt.close()

    # All families on the same plot (strict + loose separate)
    fam_list = sorted(families)

    # Prepare per-family time series
    fam_strict = {}
    fam_loose = {}
    for fam in fam_list:
        s_vals, l_vals = [], []
        for row in rows:
            rec = (row.get("by_family") or {}).get(fam)
            if rec is None:
                s_vals.append(np.nan)
                l_vals.append(np.nan)
            else:
                s_vals.append(rec.get("strict", np.nan))
                l_vals.append(rec.get("loose", np.nan))
        fam_strict[fam] = np.array(s_vals, dtype=float)
        fam_loose[fam] = np.array(l_vals, dtype=float)

    fam_list_nonempty_strict = [
        f for f in fam_list if not np.all(np.isnan(fam_strict[f]))
    ]
    fam_list_nonempty_loose = [
        f for f in fam_list if not np.all(np.isnan(fam_loose[f]))
    ]

    # Strict plot (all families)
    plt.figure()
    for fam in fam_list_nonempty_strict:
        plt.plot(xs, fam_strict[fam], marker="o", label=fam)
    plt.xlabel("Compression ratio")
    plt.ylabel("Strict")
    plt.title("All Families (Strict) vs Compression Ratio")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(
        title="Family", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0
    )
    plt.savefig(plot_path / "families_all_strict_vs_ratio.png", bbox_inches="tight")
    plt.close()

    # Loose plot (all families)
    plt.figure()
    for fam in fam_list_nonempty_loose:
        plt.plot(xs, fam_loose[fam], marker="o", label=fam)
    plt.xlabel("Compression ratio")
    plt.ylabel("Loose")
    plt.title("All Families (Loose) vs Compression Ratio")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(
        title="Family", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0
    )
    plt.savefig(plot_path / "families_all_loose_vs_ratio.png", bbox_inches="tight")
    plt.close()

    # system prompt length bins (if present)
    if "by_syslen_bins" in rows[0]:
        syslen_dir = plot_path / "syslen"
        syslen_dir.mkdir(parents=True, exist_ok=True)

        first = rows[0]["by_syslen_bins"]
        edges = first.get("edges", [])
        labels = [f"[{edges[i]},{edges[i + 1]}]" for i in range(len(edges) - 1)]

        for metric in overall_metrics:
            plt.figure()
            for b_idx, label in enumerate(labels):
                ys = []
                for row in rows:
                    bins = row["by_syslen_bins"]["bins"]
                    if b_idx < len(bins):
                        ys.append(bins[b_idx].get(metric, np.nan))
                    else:
                        ys.append(np.nan)
                plt.plot(xs, ys, marker="o", label=label)
            plt.xlabel("Compression ratio")
            plt.ylabel(metric.replace("_", " ").title())
            plt.title(
                f"{metric.replace('_', ' ').title()} vs Compression Ratio by System Length"
            )
            plt.ylim(0.0, 1.0)
            plt.grid(True, alpha=0.3)
            plt.legend(title="Sys tokens")
            plt.savefig(
                syslen_dir / f"{metric}_by_syslen_vs_ratio.png", bbox_inches="tight"
            )
            plt.close()

    # --- Two-family co-occurrence subset analysis ---
    families = [
        "change_case",
        "combination",
        "detectable_content",
        "detectable_format",
        "keywords",
        "language",
        "length_constraints",
        "punctuation",
        "startend",
    ]

    for i in range(len(families)):
        for j in range(i + 1, len(families)):
            fam_a = families[i]
            fam_b = families[j]
            plot_two_family_cooccurrence_subset(
                results_all_ratios, plot_path / "cooccur", fam_pair=(fam_a, fam_b)
            )

    # --- Single vs Multi (overall) ---
    # Collect series if available
    have_single_multi = all("single_multi" in row for row in rows)
    if have_single_multi:
        metrics = [
            ("prompt_strict", "Prompt Strict"),
            ("prompt_loose", "Prompt Loose"),
            ("instruction_strict", "Instruction Strict"),
            ("instruction_loose", "Instruction Loose"),
        ]
        for key, pretty in metrics:
            ys_single = np.array(
                [r["single_multi"]["single"].get(key, np.nan) for r in rows],
                dtype=float,
            )
            ys_multi = np.array(
                [r["single_multi"]["multi"].get(key, np.nan) for r in rows], dtype=float
            )
            err_single = np.array(
                [
                    r["single_multi"]["single"].get(
                        key.replace("instruction", "instruction_err").replace(
                            "prompt", "prompt_err"
                        ),
                        np.nan,
                    )
                    for r in rows
                ],
                dtype=float,
            )
            err_multi = np.array(
                [
                    r["single_multi"]["multi"].get(
                        key.replace("instruction", "instruction_err").replace(
                            "prompt", "prompt_err"
                        ),
                        np.nan,
                    )
                    for r in rows
                ],
                dtype=float,
            )

            plt.figure()
            # Use error bars when errors are present (non-NaN)
            if not np.all(np.isnan(err_single)) and not np.all(np.isnan(err_multi)):
                plt.errorbar(
                    xs,
                    ys_single,
                    yerr=err_single,
                    marker="o",
                    capsize=3,
                    label="single",
                )
                plt.errorbar(
                    xs, ys_multi, yerr=err_multi, marker="s", capsize=3, label="multi"
                )
            else:
                plt.plot(xs, ys_single, marker="o", label="single")
                plt.plot(xs, ys_multi, marker="s", label="multi")
            plt.xlabel("Compression ratio")
            plt.ylabel(pretty)
            plt.title(f"{pretty} vs Compression Ratio (Single vs Multi)")
            plt.ylim(0.0, 1.0)
            plt.grid(True, alpha=0.3)
            plt.legend(title="Prompt type")
            safe = key
            plt.savefig(plot_path / f"{safe}_single_vs_multi.png", bbox_inches="tight")
            plt.close()

        # --- Per-family curves for SINGLE-ONLY prompts ---
    if all(("by_family_single" in r) for r in rows):
        fam_dir_single = plot_path / "families_single"
        fam_dir_single.mkdir(parents=True, exist_ok=True)
        fams_single = set()
        for r in rows:
            fams_single |= set((r.get("by_family_single") or {}).keys())
        for fam in sorted(fams_single):
            strict_vals, loose_vals = [], []
            for r in rows:
                rec = (r.get("by_family_single") or {}).get(fam)
                if rec is None:
                    strict_vals.append(np.nan)
                    loose_vals.append(np.nan)
                else:
                    strict_vals.append(rec.get("strict", np.nan))
                    loose_vals.append(rec.get("loose", np.nan))
            plt.figure()
            plt.plot(xs, np.array(strict_vals, dtype=float), marker="o", label="strict")
            plt.plot(xs, np.array(loose_vals, dtype=float), marker="s", label="loose")
            plt.xlabel("Compression ratio")
            plt.ylabel("Accuracy")
            plt.title(f"[Single] {fam} Accuracy vs Compression Ratio")
            plt.ylim(0.0, 1.0)
            plt.grid(True, alpha=0.3)
            plt.legend()
            safe = fam.replace(":", "_").replace("/", "_")
            plt.savefig(
                fam_dir_single / f"family_{safe}_single_vs_ratio.png",
                bbox_inches="tight",
            )
            plt.close()

    # --- Per-family curves for MULTI-INSTRUCTION prompts ---
    if all(("by_family_multi" in r) for r in rows):
        fam_dir_multi = plot_path / "families_multi"
        fam_dir_multi.mkdir(parents=True, exist_ok=True)
        fams_multi = set()
        for r in rows:
            fams_multi |= set((r.get("by_family_multi") or {}).keys())
        for fam in sorted(fams_multi):
            strict_vals, loose_vals = [], []
            for r in rows:
                rec = (r.get("by_family_multi") or {}).get(fam)
                if rec is None:
                    strict_vals.append(np.nan)
                    loose_vals.append(np.nan)
                else:
                    strict_vals.append(rec.get("strict", np.nan))
                    loose_vals.append(rec.get("loose", np.nan))
            plt.figure()
            plt.plot(xs, np.array(strict_vals, dtype=float), marker="o", label="strict")
            plt.plot(xs, np.array(loose_vals, dtype=float), marker="s", label="loose")
            plt.xlabel("Compression ratio")
            plt.ylabel("Accuracy")
            plt.title(f"[Multi] {fam} Accuracy vs Compression Ratio")
            plt.ylim(0.0, 1.0)
            plt.grid(True, alpha=0.3)
            plt.legend()
            safe = fam.replace(":", "_").replace("/", "_")
            plt.savefig(
                fam_dir_multi / f"family_{safe}_multi_vs_ratio.png", bbox_inches="tight"
            )
            plt.close()

    # All families on one figure for SINGLE-only prompts (strict + loose separate)
    if all(("by_family_single" in r) for r in rows):
        fam_list_single = sorted(fams_single)  # type: ignore
        # Build per-family series
        fam_strict_single = {}
        fam_loose_single = {}
        for fam in fam_list_single:
            s_vals, l_vals = [], []
            for r in rows:
                rec = (r.get("by_family_single") or {}).get(fam)
                if rec is None:
                    s_vals.append(np.nan)
                    l_vals.append(np.nan)
                else:
                    s_vals.append(rec.get("strict", np.nan))
                    l_vals.append(rec.get("loose", np.nan))
            fam_strict_single[fam] = np.array(s_vals, dtype=float)
            fam_loose_single[fam] = np.array(l_vals, dtype=float)

        fam_list_nonempty_strict_single = [
            f for f in fam_list_single if not np.all(np.isnan(fam_strict_single[f]))
        ]
        fam_list_nonempty_loose_single = [
            f for f in fam_list_single if not np.all(np.isnan(fam_loose_single[f]))
        ]

        # Strict overlay
        plt.figure()
        for fam in fam_list_nonempty_strict_single:
            plt.plot(xs, fam_strict_single[fam], label=fam)
        plt.xlabel("Compression ratio")
        plt.ylabel("Strict")
        plt.title("All Families (Strict) vs Compression Ratio — SINGLE")
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend(
            title="Family",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
        )
        plt.savefig(
            plot_path / "families_all_strict_single_vs_ratio.png", bbox_inches="tight"
        )
        plt.close()

        # Loose overlay
        plt.figure()
        for fam in fam_list_nonempty_loose_single:
            plt.plot(xs, fam_loose_single[fam], label=fam)
        plt.xlabel("Compression ratio")
        plt.ylabel("Loose")
        plt.title("All Families (Loose) vs Compression Ratio — SINGLE")
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend(
            title="Family",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
        )
        plt.savefig(
            plot_path / "families_all_loose_single_vs_ratio.png", bbox_inches="tight"
        )
        plt.close()

    # All families on one figure for MULTI-instruction prompts (strict + loose separate)
    if all(("by_family_multi" in r) for r in rows):
        fam_list_multi = sorted(fams_multi)  # type: ignore
        # Build per-family series
        fam_strict_multi = {}
        fam_loose_multi = {}
        for fam in fam_list_multi:
            s_vals, l_vals = [], []
            for r in rows:
                rec = (r.get("by_family_multi") or {}).get(fam)
                if rec is None:
                    s_vals.append(np.nan)
                    l_vals.append(np.nan)
                else:
                    s_vals.append(rec.get("strict", np.nan))
                    l_vals.append(rec.get("loose", np.nan))
            fam_strict_multi[fam] = np.array(s_vals, dtype=float)
            fam_loose_multi[fam] = np.array(l_vals, dtype=float)

        fam_list_nonempty_strict_multi = [
            f for f in fam_list_multi if not np.all(np.isnan(fam_strict_multi[f]))
        ]
        fam_list_nonempty_loose_multi = [
            f for f in fam_list_multi if not np.all(np.isnan(fam_loose_multi[f]))
        ]

        # Strict overlay
        plt.figure()
        for fam in fam_list_nonempty_strict_multi:
            plt.plot(xs, fam_strict_multi[fam], label=fam)
        plt.xlabel("Compression ratio")
        plt.ylabel("Strict")
        plt.title("All Families (Strict) vs Compression Ratio — MULTI")
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend(
            title="Family",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
        )
        plt.savefig(
            plot_path / "families_all_strict_multi_vs_ratio.png", bbox_inches="tight"
        )
        plt.close()

        # Loose overlay
        plt.figure()
        for fam in fam_list_nonempty_loose_multi:
            plt.plot(xs, fam_loose_multi[fam], label=fam)
        plt.xlabel("Compression ratio")
        plt.ylabel("Loose")
        plt.title("All Families (Loose) vs Compression Ratio — MULTI")
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend(
            title="Family",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
        )
        plt.savefig(
            plot_path / "families_all_loose_multi_vs_ratio.png", bbox_inches="tight"
        )
        plt.close()


def most_common_family_pair(
    results_all_ratios: dict[str, dict],
) -> tuple[str, str] | None:
    """Scan per-prompt data across all ratios and return the most common two-family co-occurrence."""
    counts = {}
    for res in results_all_ratios.values():
        per_prompt = res.get("per_prompt") or []
        for rec in per_prompt:
            fams = sorted(set(rec.get("families", [])))
            for a, b in combinations(fams, 2):
                counts[(a, b)] = counts.get((a, b), 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def plot_two_family_cooccurrence_subset(
    results_all_ratios: dict[str, dict],
    plot_path: Path,
    fam_pair: tuple[str, str] | None = None,
    title_suffix: str = "",
):
    """
    Restrict to prompts that contain BOTH families in fam_pair.
    Plot strict and loose accuracy for just those two families across ratios.
    """
    plot_path.mkdir(parents=True, exist_ok=True)

    # Sort ratios
    ratio_keys = sorted(
        results_all_ratios.keys(), key=lambda k: float(k.split("_", 1)[1])
    )
    xs = [float(k.split("_", 1)[1]) for k in ratio_keys]
    rows = [results_all_ratios[k] for k in ratio_keys]

    # Determine pair if not provided
    if fam_pair is None:
        fam_pair = most_common_family_pair(results_all_ratios)
    if fam_pair is None:
        # No co-occurrence available; nothing to plot
        return
    fam_a, fam_b = fam_pair

    # Build series
    strict_a, strict_b, loose_a, loose_b = [], [], [], []

    for row in rows:
        per_prompt = row.get("per_prompt") or []
        # Gather all instructions (indices) in prompts that contain BOTH families
        a_hits_strict = []
        b_hits_strict = []
        a_hits_loose = []
        b_hits_loose = []

        for rec in per_prompt:
            fams = rec["families"]
            if fam_a in fams and fam_b in fams:
                # For each instruction in this prompt, bucket by its family
                s_list = rec["strict_list"]
                l_list = rec["loose_list"]
                for inst_fam, s_ok, l_ok in zip(fams, s_list, l_list):
                    if inst_fam == fam_a:
                        a_hits_strict.append(bool(s_ok))
                        a_hits_loose.append(bool(l_ok))
                    elif inst_fam == fam_b:
                        b_hits_strict.append(bool(s_ok))
                        b_hits_loose.append(bool(l_ok))

        # Compute means (NaN if empty)
        def mean_or_nan(arr):
            return float(np.mean(arr)) if len(arr) > 0 else np.nan

        strict_a.append(mean_or_nan(a_hits_strict))
        strict_b.append(mean_or_nan(b_hits_strict))
        loose_a.append(mean_or_nan(a_hits_loose))
        loose_b.append(mean_or_nan(b_hits_loose))

    # Plot STRICT (two lines)
    plt.figure()
    plt.plot(xs, np.array(strict_a, dtype=float), marker="o", label=fam_a)
    plt.plot(xs, np.array(strict_b, dtype=float), marker="s", label=fam_b)
    plt.xlabel("Compression ratio")
    plt.ylabel("Strict")
    ttl = f"Co-occurrence: {fam_a} vs {fam_b} (Strict)"
    if title_suffix:
        ttl += f" — {title_suffix}"
    plt.title(ttl)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Family")
    safe_a = fam_a.replace(":", "_").replace("/", "_")
    safe_b = fam_b.replace(":", "_").replace("/", "_")
    plt.savefig(
        plot_path / f"cooccur_{safe_a}_vs_{safe_b}_strict_vs_ratio.png",
        bbox_inches="tight",
    )
    plt.close()

    # Plot LOOSE (two lines)
    plt.figure()
    plt.plot(xs, np.array(loose_a, dtype=float), marker="o", label=fam_a)
    plt.plot(xs, np.array(loose_b, dtype=float), marker="s", label=fam_b)
    plt.xlabel("Compression ratio")
    plt.ylabel("Loose")
    ttl = f"Co-occurrence: {fam_a} vs {fam_b} (Loose)"
    if title_suffix:
        ttl += f" — {title_suffix}"
    plt.title(ttl)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Family")
    plt.savefig(
        plot_path / f"cooccur_{safe_a}_vs_{safe_b}_loose_vs_ratio.png",
        bbox_inches="tight",
    )
    plt.close()

    # Find how many prompts were used for co-occurrence at the first ratio. Should be consistent across ratios.
    n_prompts_used = 0
    first_row = rows[0]
    if first_row:
        per_prompt = first_row.get("per_prompt") or []
        for rec in per_prompt:
            fams = rec.get("families", [])
            if fam_a in fams and fam_b in fams:
                n_prompts_used += 1

    logger = logging.getLogger(__name__)
    logger.info(f"Co-occurrence {fam_a} & {fam_b}: {n_prompts_used} prompts")


def main(config: CompressedContextIFEval):
    """
    1. Read IFEval dataset (sys_ifeval.jsonl)
    2. For each prompt, compress the context using the specified press strategy and compression ratios
    3. Save the responses into a jsonl file
    4. Evaluate the model's performance on the compressed context using IFEval metrics. Requires the response json file.

    Note: ifeval --> sys_ifeval. For brevity, we use "ifeval" in place of "sys_ifeval" in variable names and function names.
    """
    set_seeds(config.seed)

    if config.analyze_kept_tokens:
        assert config.defense_template_key is not None, (
            "Kept token analysis requires a defense template to be applied."
        )

    # Create output directories
    config.outputs_base.mkdir(parents=True, exist_ok=True)
    run_name_defense_template_part = (
        f"_defense_{config.defense_template_key}" if config.defense_template_key else ""
    )
    run_name_default = f"{config.model_name_shorthand}_{config.press_name}{run_name_defense_template_part}_num_prompts_{config.num_prompts}_ratio_start_{config.compression_ratio_start}_ratio_end_{config.compression_ratio_end}_ratio_steps_{config.compression_ratio_steps}_max_new_tokens_{config.max_new_tokens}_sample_responses_{config.sample_responses}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = config.run_name if config.run_name else run_name_default
    run_path = config.outputs_base / run_dir
    run_path.mkdir(parents=True, exist_ok=True)
    log_path = run_path / "logs"
    log_path.mkdir(parents=True, exist_ok=True)
    responses_path = run_path / "responses"
    responses_path.mkdir(parents=True, exist_ok=True)
    plot_path = run_path / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        filename=log_path / "evaluation.log",
        level=getattr(logging, config.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Config: {config}")

    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=get_attention_implementation_from_strategy(
            config.press_name
        ),
        device_map="auto",
        cache_dir=config.model_cache_dir,
        trust_remote_code=True,
    )
    model.config.output_attentions = get_output_attentions_from_strategy(
        config.press_name
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        cache_dir=config.model_cache_dir,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    logger.info("Model and tokenizer loaded successfully.")

    ifeval_items = read_jsonl(config.sys_ifeval_path, config.num_prompts)
    logger.info(f"Read {len(ifeval_items)} prompt items from IFEval dataset.")
    logger.debug(f"Prompt items: {ifeval_items}")

    ratio_range = torch.linspace(
        config.compression_ratio_start,
        config.compression_ratio_end,
        steps=config.compression_ratio_steps,
    )
    logger.info(f"Compression ratios: {ratio_range.tolist()}")
    press = press_resolver(config.press_name)
    press.force_keep_global = (
        torch.tensor(config.force_keep_global)
        if config.force_keep_global is not None
        else None
    )
    if config.defense_span is not None:
        if hasattr(press, "defense_span"):
            press.defense_span = config.defense_span  # type: ignore
            logger.info(
                f"Using defense span: {press.defense_span} in SystemPromptPress"  # type: ignore
            )

    if config.system_prompt_prepend is not None:
        # Prepend to each system prompt
        for item in ifeval_items:
            original_messages = item["messages"]
            assert original_messages[0]["role"] == "system", (
                "First message must be system prompt"
            )
            system_prompt = original_messages[0]["content"]
            new_system_prompt = config.system_prompt_prepend + system_prompt
            new_messages = [
                {"role": "system", "content": new_system_prompt}
            ] + original_messages[1:]
            item["messages"] = new_messages

    if config.defense_template_key is not None:
        # Load defense template
        with open(config.defense_template_path, "r", encoding="utf-8") as f:
            template_json = json.load(f)
        defense_template = template_json[config.defense_template_key]
        logger.info(
            f"Loaded defense template for key {config.defense_template_key} from {config.defense_template_path}"
        )
        logger.info(f"Defense template: {defense_template}")

        # Apply defense template to each prompt in ifeval_items
        for item in ifeval_items:
            original_messages = item["messages"]
            assert original_messages[0]["role"] == "system", (
                "First message must be system prompt"
            )
            system_prompt = original_messages[0]["content"]
            defended_system_prompt = defense_template.replace(
                "$system_prompt", system_prompt
            )
            new_messages = [
                {"role": "system", "content": defended_system_prompt}
            ] + original_messages[1:]
            item["messages"] = new_messages
            item["original_messages"] = original_messages
            item["defense_template"] = defense_template

    ifeval_results_all_ratios = {}
    for ratio in tqdm(ratio_range, desc="Processing compression ratios"):
        logger.info(f"Processing compression ratio: {ratio.item():.2f}")
        press.compression_ratio = ratio.item()
        ratio_responses_path = responses_path / f"ratio_{ratio.item():.2f}"
        ratio_responses_path.mkdir(parents=True, exist_ok=True)
        ratio_responses_file = ratio_responses_path / "responses.jsonl"
        ratio_ifeval_results_file = ratio_responses_path / "ifeval_results.json"

        logger.info(f"Generating responses for ratio {ratio.item():.2f}...")
        responses = get_sys_ifeval_responses_at_ratio(
            ratio.item(),
            ifeval_items,
            model,
            tokenizer,
            press,
            config,
        )

        save_responses(ratio_responses_file, responses)
        logger.info(f"Saved responses to {ratio_responses_file}")

        ifeval_results = get_ifeval_results(
            input_path=config.sys_ifeval_path,
            response_path=ratio_responses_file,
            skip_ids=config.skip_ids,
            tokenizer=tokenizer,
        )
        save_ifeval_results(ratio_ifeval_results_file, ifeval_results)
        logger.info(f"Saved IFEval results to {ratio_ifeval_results_file}")

        ifeval_results_all_ratios[f"ratio_{ratio.item():.2f}"] = ifeval_results

    plot_results_across_ratios(
        ifeval_results_all_ratios,
        plot_path,
    )


if __name__ == "__main__":
    main(CLI(CompressedContextIFEval, as_positional=False))
