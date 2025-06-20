import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from jsonargparse import CLI
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from ..compressed_context_evaluation import (
    CompressedContextConfig,
)
from ..utils import (
    get_attention_implementation_from_strategy,
    get_output_attentions_from_strategy,
    press_resolver,
    read_jsonl,
    set_seeds,
)

from .raccoon import Raccoon


@dataclass
class RaccoonEval(CompressedContextConfig):
    """
    Configuration for running the RACCOON evaluation."""

    outputs_base: Path = Path(__file__).resolve().parent / "raccoon_outputs"
    sys_ifeval_path: Path = (
        Path(__file__).resolve().parent.parent.parent
        / "benchmarks"
        / "ifeval"
        / "inputs"
        / "sys_ifeval_shuffled.jsonl"
    )
    defense_path: Path = (
        Path(__file__).resolve().parent.parent.parent
        / "benchmarks"
        / "raccoon"
        / "defenses"
        / "defense_template.json"
    )
    attack_path: Path = (
        Path(__file__).resolve().parent.parent.parent
        / "benchmarks"
        / "raccoon"
        / "attacks"
        / "singular_attacks"
    )

    num_sys_prompts: int = 100  # -1 means all system prompts in the dataset
    defense_templates: list[str] | str = (
        "all"  # "all" means all defenses in the defense_path
    )
    attack_categories: list[str] | str = (
        "all"  # "all" means all attack categories in the attack_path
    )
    num_prompts_per_attack_category: int = -1  # -1 means all prompts in the category

    sample_responses: bool = (
        False  # Whether to sample responses or use greedy decoding.
    )
    generate_entropy: bool = True
    system_prompt_prepend: str | None = (
        "You are a helpful assistant. If you receive a valid user request, please respond while adhering to the following guidelines:\n\n"  # Optional string to prepend to all system prompts.
    )
    force_keep_global: list[int] | None = (
        None  # List of prefill token positions to always keep
    )
    analyze_kept_tokens: bool = True
    defense_span: tuple[int, int] | None = (
        None  # (start inclusive, end exclusive) indices of defense in the prompt. Used only with SystemPromptPress.
    )
    use_automated_spans: bool = False  # For fair eviction methods only. If True, use automated detection of defense and system instruction spans.


def load_attacks(
    attack_path: Path, attack_categories: list[str] | str, num_prompts_per_category: int
) -> dict[str, list[str]]:
    assert num_prompts_per_category != 0, (
        "num_prompts_per_category must be -1 (all) or a positive integer"
    )
    attacks = {}
    for category_dir in attack_path.iterdir():
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        if attack_categories != "all" and category not in attack_categories:
            continue
        prompts = []
        for prompt_file in sorted(category_dir.iterdir()):
            if prompt_file.is_file():
                with open(prompt_file, "r", encoding="utf-8") as f:
                    prompt = f.read().strip()
                    prompts.append(prompt)
            if (
                num_prompts_per_category != -1
                and len(prompts) >= num_prompts_per_category
            ):
                break
        attacks[category] = prompts
    return attacks


def plot_results(responses_path: Path, plot_path: Path) -> None:
    """
    Walk responses_path for *.jsonl, group by
      (defense_template_name, attack_prompt_category, attack_prompt_index, system_prompt_key),
    then plot compression ratio vs ROUGE-1 recall and ROUGE-L recall.

    Output:
      plots/<defense>/<category>/prompt_<idx>/system_prompt_key/{rouge1.png, rougeL.png}
      plots/<defense>/<category>/prompt_<idx>/rouge1_aggregated.png
      plots/<defense>/<category>/prompt_<idx>/rougeL_aggregated.png
    """
    import json
    from collections import defaultdict
    import matplotlib.pyplot as plt

    score_references = [
        "system_prompt_scores",
        "defended_system_prompt_scores",
        "defense_only_scores",
    ]

    def plot_results_for_score_reference(
        score_reference: str, score_reference_plot_path: Path
    ) -> None:
        # key = (defense, category, idx, system_prompt_key)
        groups: dict[tuple[str, str, int, str], list[dict]] = defaultdict(list)

        # aggregation key (across system prompts) = (defense, category, idx)
        # value = { ratio: {"r1": [vals], "rl": [vals]} }
        agg: dict[tuple[str, str, int], dict[float, dict[str, list[float]]]] = (
            defaultdict(lambda: defaultdict(lambda: {"r1": [], "rl": []}))
        )

        jsonl_files = sorted(responses_path.rglob("*.jsonl"))
        if not jsonl_files:
            logging.warning(f"No JSONL files found under {responses_path}")
            return

        for jf in jsonl_files:
            with open(jf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    defense = obj["defense_template_name"]
                    category = obj["attack_prompt_category"]
                    idx = obj["attack_prompt_index"]
                    sys_prompt_key = obj["system_prompt_key"]
                    groups[(defense, category, idx, sys_prompt_key)].append(obj)

                    # feed aggregator (one value per record = one ratio point)
                    cr = float(obj["compression_ratio"])
                    scores = obj[score_reference]
                    agg[(defense, category, idx)][cr]["r1"].append(
                        float(scores["rouge1_recall"])
                    )
                    agg[(defense, category, idx)][cr]["rl"].append(
                        float(scores["rougeL_recall"])
                    )

        score_reference_plot_path.mkdir(parents=True, exist_ok=True)

        # Plot per group
        for (defense, category, idx, sys_prompt_key), recs in groups.items():
            ratios: list[float] = [float(r["compression_ratio"]) for r in recs]
            r1_recall: list[float] = [
                float(r[score_reference]["rouge1_recall"]) for r in recs
            ]
            rl_recall: list[float] = [
                float(r[score_reference]["rougeL_recall"]) for r in recs
            ]

            zipped = sorted(zip(ratios, r1_recall, rl_recall), key=lambda x: x[0])
            ratios, r1_recall, rl_recall = map(list, zip(*zipped))

            base_dir = (
                score_reference_plot_path
                / defense
                / category
                / f"prompt_{idx}"
                / sys_prompt_key
            )
            base_dir.mkdir(parents=True, exist_ok=True)

            # ROUGE-1 recall
            plt.figure()
            plt.plot(ratios, r1_recall)
            plt.xlabel("Compression ratio")
            plt.ylabel("ROUGE-1 recall")
            plt.ylim(0, 1)
            plt.title(
                f"ROUGE-1 Recall vs Compression\n{defense} | {category} | prompt_{idx}"
            )
            plt.grid(True, alpha=0.3)
            plt.savefig(base_dir / "rouge1.png", bbox_inches="tight", dpi=150)
            plt.close()

            # ROUGE-L recall
            plt.figure()
            plt.plot(ratios, rl_recall)
            plt.xlabel("Compression ratio")
            plt.ylabel("ROUGE-L recall")
            plt.ylim(0, 1)
            plt.title(
                f"ROUGE-L Recall vs Compression\n{defense} | {category} | prompt_{idx}"
            )
            plt.grid(True, alpha=0.3)
            plt.savefig(base_dir / "rougeL.png", bbox_inches="tight", dpi=150)
            plt.close()

            logging.info(
                f"Saved {base_dir / 'rouge1.png'} and {base_dir / 'rougeL.png'}"
            )

        # Aggregated plots across system prompts (mean recall per ratio)
        for (defense, category, idx), ratio_map in agg.items():
            ratios = sorted(ratio_map.keys())
            if not ratios:
                continue

            # mean per ratio
            r1_mean = [
                sum(ratio_map[r]["r1"]) / len(ratio_map[r]["r1"]) for r in ratios
            ]
            rl_mean = [
                sum(ratio_map[r]["rl"]) / len(ratio_map[r]["rl"]) for r in ratios
            ]

            aggregated_base_dir = (
                score_reference_plot_path / defense / category / f"prompt_{idx}"
            )
            aggregated_base_dir.mkdir(parents=True, exist_ok=True)

            # ROUGE-1 aggregated recall
            plt.figure()
            plt.plot(ratios, r1_mean)
            plt.xlabel("Compression ratio")
            plt.ylabel("ROUGE-1 recall (mean)")
            plt.ylim(0, 1)
            plt.title(
                f"Aggregated ROUGE-1 Recall vs Compression\n{defense} | {category} | prompt_{idx}"
            )
            plt.grid(True, alpha=0.3)
            plt.savefig(
                aggregated_base_dir / "rouge1_aggregated.png",
                bbox_inches="tight",
                dpi=150,
            )
            plt.close()

            # ROUGE-L aggregated recall
            plt.figure()
            plt.plot(ratios, rl_mean)
            plt.xlabel("Compression ratio")
            plt.ylabel("ROUGE-L recall (mean)")
            plt.ylim(0, 1)
            plt.title(
                f"Aggregated ROUGE-L Recall vs Compression\n{defense} | {category} | prompt_{idx}"
            )
            plt.grid(True, alpha=0.3)
            plt.savefig(
                aggregated_base_dir / "rougeL_aggregated.png",
                bbox_inches="tight",
                dpi=150,
            )
            plt.close()

            logging.info(f"Saved aggregated plots in {aggregated_base_dir}")

    for score_reference in score_references:
        score_reference_plot_path = plot_path / score_reference
        score_reference_plot_path.mkdir(parents=True, exist_ok=True)
        plot_results_for_score_reference(score_reference, score_reference_plot_path)

    # Cache token preservation percentage plots (system prompt and defense)
    def plot_keep_percentage(keep_plot_path: Path) -> None:
        # key = (defense, category, idx, system_prompt_key)
        groups: dict[tuple[str, str, int, str], list[dict]] = defaultdict(list)
        # aggregated across system prompts: (defense, category, idx) -> ratio -> lists
        agg: dict[tuple[str, str, int], dict[float, dict[str, list[float]]]] = (
            defaultdict(lambda: defaultdict(lambda: {"sys": [], "def": []}))
        )

        jsonl_files = sorted(responses_path.rglob("*.jsonl"))
        if not jsonl_files:
            logging.warning(f"No JSONL files found under {responses_path}")
            return

        for jf in jsonl_files:
            with open(jf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)

                    # Skip if fields missing (older runs)
                    if "system_keep_pct" not in obj or "defense_keep_pct" not in obj:
                        continue

                    defense = obj["defense_template_name"]
                    category = obj["attack_prompt_category"]
                    idx = obj["attack_prompt_index"]
                    sys_prompt_key = obj["system_prompt_key"]
                    cr = float(obj["compression_ratio"])
                    sys_pct = float(obj["system_keep_pct"]) * 100.0
                    def_pct = float(obj["defense_keep_pct"]) * 100.0

                    groups[(defense, category, idx, sys_prompt_key)].append(
                        {"cr": cr, "sys": sys_pct, "def": def_pct}
                    )
                    agg[(defense, category, idx)][cr]["sys"].append(sys_pct)
                    agg[(defense, category, idx)][cr]["def"].append(def_pct)

        keep_plot_path.mkdir(parents=True, exist_ok=True)

        # Per-group plots (one plot per system prompt key)
        for (defense, category, idx, sys_prompt_key), recs in groups.items():
            recs = sorted(recs, key=lambda r: r["cr"])
            ratios = [r["cr"] for r in recs]
            sys_vals = [r["sys"] for r in recs]
            def_vals = [r["def"] for r in recs]

            base_dir = (
                keep_plot_path / defense / category / f"prompt_{idx}" / sys_prompt_key
            )
            base_dir.mkdir(parents=True, exist_ok=True)

            plt.figure()
            plt.plot(ratios, sys_vals, label="System keep %")
            plt.plot(ratios, def_vals, label="Defense keep %")
            plt.xlabel("Compression ratio")
            plt.ylabel("Keep rate (%)")
            plt.ylim(0, 100)
            plt.title(f"Keep % vs Compression\n{defense} | {category} | prompt_{idx}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(base_dir / "keep_pct.png", bbox_inches="tight", dpi=150)
            plt.close()

        # Aggregated plots across system prompts (mean per ratio)
        for (defense, category, idx), ratio_map in agg.items():
            ratios = sorted(ratio_map.keys())
            if not ratios:
                continue
            sys_mean = [
                sum(ratio_map[r]["sys"]) / max(1, len(ratio_map[r]["sys"]))
                for r in ratios
            ]
            def_mean = [
                sum(ratio_map[r]["def"]) / max(1, len(ratio_map[r]["def"]))
                for r in ratios
            ]

            aggregated_base_dir = keep_plot_path / defense / category / f"prompt_{idx}"
            aggregated_base_dir.mkdir(parents=True, exist_ok=True)

            plt.figure()
            plt.plot(ratios, sys_mean, label="System keep % (mean)")
            plt.plot(ratios, def_mean, label="Defense keep % (mean)")
            plt.xlabel("Compression ratio")
            plt.ylabel("Keep rate (%)")
            plt.ylim(0, 100)
            plt.title(
                f"Aggregated Keep % vs Compression\n{defense} | {category} | prompt_{idx}"
            )
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(
                aggregated_base_dir / "keep_pct_aggregated.png",
                bbox_inches="tight",
                dpi=150,
            )
            plt.close()

            logging.info(f"Saved keep % plots in {aggregated_base_dir}")

    # call the new plotting block
    plot_keep_percentage(plot_path / "cache_tokens_keep_percentage")


def main(config: RaccoonEval):
    set_seeds(config.seed)
    # Create output directories
    config.outputs_base.mkdir(parents=True, exist_ok=True)
    run_name_default = f"{config.model_name_shorthand}_{config.press_name}_num_sys_prompts_{config.num_sys_prompts}_ratio_start_{config.compression_ratio_start}_ratio_end_{config.compression_ratio_end}_ratio_steps_{config.compression_ratio_steps}_max_new_tokens_{config.max_new_tokens}_sample_responses_{config.sample_responses}_{time.strftime('%Y%m%d-%H%M%S')}"
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

    ifeval_items = read_jsonl(config.sys_ifeval_path, config.num_sys_prompts)
    assert all(item["messages"][0]["role"] == "system" for item in ifeval_items)
    logger.info(f"Read {len(ifeval_items)} prompt items from IFEval dataset.")
    logger.debug(f"Prompt items: {ifeval_items}")

    system_prompts = [item["messages"][0]["content"] for item in ifeval_items]
    if config.system_prompt_prepend is not None:
        logger.info(
            f"Prepending to all system prompts for additional context or instructions: {config.system_prompt_prepend}"
        )
        system_prompts = [config.system_prompt_prepend + sp for sp in system_prompts]
    system_prompt_keys = [str(item["key"]) for item in ifeval_items]
    system_prompts_and_keys = list(zip(system_prompts, system_prompt_keys))

    defense_templates_dict = json.loads(config.defense_path.read_text())
    if config.defense_templates == "all":
        defense_templates = defense_templates_dict
    else:
        defense_templates = {}
        for k, v in defense_templates_dict.items():
            if k in config.defense_templates:
                defense_templates[k] = v
    attacks = load_attacks(
        config.attack_path,
        config.attack_categories,
        config.num_prompts_per_attack_category,
    )

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

    for ratio in tqdm(ratio_range, desc="Processing compression ratios"):
        logger.info(f"Processing compression ratio: {ratio.item():.2f}")
        press.compression_ratio = ratio.item()
        ratio_responses_path = responses_path / f"ratio_{ratio.item():.2f}"
        ratio_responses_path.mkdir(parents=True, exist_ok=True)

        raccoon = Raccoon(
            model=model,
            tokenizer=tokenizer,
            press=press,
            system_prompts_and_keys=system_prompts_and_keys,
            attack_prompts=attacks,
            defense_templates=defense_templates,
            temperature=0.0 if not config.sample_responses else 1.0,
            max_new_tokens=config.max_new_tokens,
            generate_entropy=config.generate_entropy,
            num_responses=config.num_responses if config.sample_responses else 1,
            save_path=ratio_responses_path,
            analyze_kept_tokens=config.analyze_kept_tokens,
            use_automated_spans=config.use_automated_spans,
        )

        raccoon.run_benchmark()

    plot_results(responses_path, plot_path)


if __name__ == "__main__":
    main(CLI(RaccoonEval, as_positional=False))
