import logging
from dataclasses import dataclass
from pathlib import Path

from jsonargparse import CLI
import json
from collections import defaultdict
import matplotlib.pyplot as plt


@dataclass
class RaccoonCombinedPlotterConfig:
    response_paths: list[Path]
    run_name: str
    outputs_base: Path = Path(__file__).parent / "raccoon_combined_plotter_outputs"
    log_level: str = "info"


def _label_for_method(responses_path: Path) -> str:
    """
    Derive a legend label from the run_dir that contains the method token.
    Supports these method tokens (case-insensitive):
      - streamingllm
      - observed_attention
      - pyramid
      - snap
      - tova
      - knorm

    Expected path shape (either is fine):
      .../<run_dir>/responses
      .../<run_dir>
    """
    method_map = {
        "streamingllm": "StreamingLLM",
        "observed_attention": "Observed Attention",
        "pyramid": "Pyramid",
        "snap": "SnapKV",
        "tova": "TOVA",
        "knorm": "Knorm",
    }

    # Identify run_dir from the provided path
    run_dir = (
        responses_path.parent.name
        if responses_path.name == "responses"
        else responses_path.name
    )
    lower = run_dir.lower()

    # Fast pattern checks to avoid substring collisions
    def has_token(tok: str) -> bool:
        # Allow token as its own segment or at string edges
        return (
            f"_{tok}_" in lower
            or lower.endswith(f"_{tok}")
            or lower.startswith(f"{tok}_")
            or lower == tok
            or tok in lower.split("_")
        )

    for token, label in method_map.items():
        if has_token(token):
            return label

    logging.warning(
        f"Could not detect compression method from run dir '{run_dir}'. Using run dir as label."
    )
    return run_dir


def plot_combined_results(response_paths, plot_path):
    logger = logging.getLogger(__name__)

    score_references = [
        "system_prompt_scores",
        "defended_system_prompt_scores",
        "defense_only_scores",
    ]

    # method_means[method_label][(defense, category, idx)][score_ref] -> {
    #   "ratios": [sorted float],
    #   "r1":     [mean recall per ratio],
    #   "rl":     [mean recall per ratio],
    # }
    method_means = {}

    # Keep union of keys so we know what to plot across methods
    union_keys_per_score_ref = {sr: set() for sr in score_references}

    for rp in response_paths:
        method_label = _label_for_method(Path(rp))
        logger.info(f"Processing method: {method_label} from {rp}")

        jsonl_files = sorted(Path(rp).rglob("*.jsonl"))
        if not jsonl_files:
            logger.warning(f"No JSONL files found under {rp}")
            continue
        logger.info(f"Found {len(jsonl_files)} JSONL files under {rp}")

        # agg[sr][(defense, category, idx)][ratio] -> {"r1": [vals], "rl": [vals]}
        agg = {
            sr: defaultdict(lambda: defaultdict(lambda: {"r1": [], "rl": []}))
            for sr in score_references
        }

        for jf in jsonl_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)

                        defense = obj.get("defense_template_name")
                        category = obj.get("attack_prompt_category")
                        idx = obj.get("attack_prompt_index")
                        if defense is None or category is None or idx is None:
                            continue

                        try:
                            cr = float(obj["compression_ratio"])
                        except Exception:
                            logger.warning(f"Missing/invalid compression_ratio in {jf}")
                            continue

                        for sr in score_references:
                            scores = obj.get(sr)
                            if not scores:
                                # some records may not include all three refs
                                continue
                            try:
                                r1 = float(scores["rouge1_recall"])
                                rl = float(scores["rougeL_recall"])
                            except Exception:
                                logger.warning(f"Missing rouge keys for {sr} in {jf}")
                                continue

                            agg[sr][(defense, category, idx)][cr]["r1"].append(r1)
                            agg[sr][(defense, category, idx)][cr]["rl"].append(rl)
            except Exception as e:
                logger.exception(f"Failed to parse {jf}: {e}")

        # Convert to means
        method_means[method_label] = {}
        for sr in score_references:
            for key, ratio_map in agg[sr].items():
                ratios_sorted = sorted(ratio_map.keys())
                if not ratios_sorted:
                    continue
                r1_mean = [
                    sum(ratio_map[r]["r1"]) / max(1, len(ratio_map[r]["r1"]))
                    for r in ratios_sorted
                ]
                rl_mean = [
                    sum(ratio_map[r]["rl"]) / max(1, len(ratio_map[r]["rl"]))
                    for r in ratios_sorted
                ]
                if key not in method_means[method_label]:
                    method_means[method_label][key] = {}
                method_means[method_label][key][sr] = {
                    "ratios": ratios_sorted,
                    "r1": r1_mean,
                    "rl": rl_mean,
                }
                union_keys_per_score_ref[sr].add(key)

        logger.info(f"Finished aggregating method: {method_label}")

    # Plot combined overlays (4 lines) per (score_ref, defense, category, idx)
    for sr in score_references:
        sr_plot_root = Path(plot_path) / sr
        for defense, category, idx in sorted(union_keys_per_score_ref[sr]):
            base_dir = sr_plot_root / defense / category / f"prompt_{idx}"
            base_dir.mkdir(parents=True, exist_ok=True)

            # ROUGE-1
            plt.figure()
            any_plotted = False
            for method_label, method_dict in method_means.items():
                if (defense, category, idx) not in method_dict:
                    continue
                if sr not in method_dict[(defense, category, idx)]:
                    continue
                series = method_dict[(defense, category, idx)][sr]
                plt.plot(series["ratios"], series["r1"], label=method_label)
                any_plotted = True

            if any_plotted:
                plt.xlabel("Compression ratio")
                plt.ylabel("ROUGE-1 recall (mean)")
                plt.ylim(0, 1)
                plt.title(
                    f"ROUGE-1 Recall vs Compression (combined)\n{defense} | {category} | prompt_{idx}"
                )
                plt.grid(True, alpha=0.3)
                plt.legend()
                out_path = base_dir / "rouge1_combined.png"
                plt.savefig(out_path, bbox_inches="tight", dpi=150)
                plt.close()
                logger.info(f"Saved {out_path}")
            else:
                logger.warning(
                    f"No data to plot for ROUGE-1: {sr} | {defense} | {category} | prompt_{idx}"
                )

            # ROUGE-L
            plt.figure()
            any_plotted = False
            for method_label, method_dict in method_means.items():
                if (defense, category, idx) not in method_dict:
                    continue
                if sr not in method_dict[(defense, category, idx)]:
                    continue
                series = method_dict[(defense, category, idx)][sr]
                plt.plot(series["ratios"], series["rl"], label=method_label)
                any_plotted = True

            if any_plotted:
                plt.xlabel("Compression ratio")
                plt.ylabel("ROUGE-L recall (mean)")
                plt.ylim(0, 1)
                plt.title(
                    f"ROUGE-L Recall vs Compression (combined)\n{defense} | {category} | prompt_{idx}"
                )
                plt.grid(True, alpha=0.3)
                plt.legend()
                out_path = base_dir / "rougeL_combined.png"
                plt.savefig(out_path, bbox_inches="tight", dpi=150)
                plt.close()
                logger.info(f"Saved {out_path}")
            else:
                logger.warning(
                    f"No data to plot for ROUGE-L: {sr} | {defense} | {category} | prompt_{idx}"
                )


def main(config: RaccoonCombinedPlotterConfig):
    # Create output directories
    run_dir = config.run_name
    run_path = config.outputs_base / run_dir
    run_path.mkdir(parents=True, exist_ok=True)
    plot_path = run_path / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)
    log_path = run_path / "logs"
    log_path.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        filename=log_path / "evaluation.log",
        level=getattr(logging, config.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    plot_combined_results(config.response_paths, plot_path)


if __name__ == "__main__":
    main(CLI(RaccoonCombinedPlotterConfig, as_positional=False))
