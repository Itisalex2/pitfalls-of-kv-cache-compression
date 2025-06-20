import csv
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from jsonargparse import CLI

from .run_raccoon import plot_results


@dataclass
class RaccoonPlotterConfig:
    response_path: Path
    outputs_base: Path = Path(__file__).parent / "raccoon_plotter_outputs"
    run_name: str = ""  # Custom run name, if empty will be generated automatically
    log_level: str = "info"
    output_csv: bool = False


def _sem(vals: list[float]) -> float:
    """Standard error of the mean with ddof=1; returns 0.0 for n<=1."""
    nums = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    n = len(nums)
    if n <= 1:
        return 0.0
    mean = sum(nums) / n
    var = sum((x - mean) ** 2 for x in nums) / (n - 1)
    return (var**0.5) / (n**0.5)


def save_raccoon_csvs(responses_path: Path, csv_base_dir: Path) -> None:
    """
    Aggregate per (defense, category, prompt_idx, score_reference) across system prompts,
    computing mean and SEM at each compression ratio, and save CSVs.

    Output tree:
      csv/<score_reference>/<defense>/<category>/prompt_<idx>/results.csv
    """

    score_references = [
        "system_prompt_scores",
        "defended_system_prompt_scores",
        "defense_only_scores",
    ]

    # data[(defense, category, idx, score_ref)][ratio] = {
    #   "r1": [vals], "rl": [vals], "sys": [keep%], "def": [keep%]
    # }
    data: dict[
        tuple[str, str, int, str],
        dict[float, dict[str, list[float]]],
    ] = defaultdict(
        lambda: defaultdict(lambda: {"r1": [], "rl": [], "sys": [], "def": []})
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
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                # Required keys
                if not all(
                    k in obj
                    for k in (
                        "defense_template_name",
                        "attack_prompt_category",
                        "attack_prompt_index",
                        "compression_ratio",
                    )
                ):
                    continue

                defense = obj["defense_template_name"]
                category = obj["attack_prompt_category"]
                idx = int(obj["attack_prompt_index"])
                cr = float(obj["compression_ratio"])

                # Optional keep% (may not exist on older runs)
                sys_pct = obj.get("system_keep_pct", None)
                def_pct = obj.get("defense_keep_pct", None)
                if isinstance(sys_pct, (int, float)):
                    # your stored value is fraction; store as % for CSV
                    sys_pct = float(sys_pct) * 100.0
                if isinstance(def_pct, (int, float)):
                    def_pct = float(def_pct) * 100.0

                for score_ref in score_references:
                    if score_ref not in obj:
                        continue
                    scores = obj[score_ref] or {}
                    r1 = scores.get("rouge1_recall", None)
                    rl = scores.get("rougeL_recall", None)

                    key = (defense, category, idx, score_ref)
                    if isinstance(r1, (int, float)):
                        data[key][cr]["r1"].append(float(r1))
                    if isinstance(rl, (int, float)):
                        data[key][cr]["rl"].append(float(rl))
                    # keep% is independent of score_ref but we copy into each for convenience
                    if isinstance(sys_pct, (int, float)):
                        data[key][cr]["sys"].append(float(sys_pct))
                    if isinstance(def_pct, (int, float)):
                        data[key][cr]["def"].append(float(def_pct))

    # Write CSVs
    for (defense, category, idx, score_ref), ratio_map in data.items():
        ratios = sorted(ratio_map.keys())
        out_dir = (
            csv_base_dir / "csv" / score_ref / defense / category / f"prompt_{idx}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "results.csv"

        headers = [
            "compression_ratio",
            "rouge1",
            "rouge1_err",
            "rougeL",
            "rougeL_err",
            "system_keep_pct",
            "system_keep_pct_err",
            "defense_keep_pct",
            "defense_keep_pct_err",
        ]
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(headers)
            for r in ratios:
                buckets = ratio_map[r]
                r1_vals = buckets.get("r1", [])
                rl_vals = buckets.get("rl", [])
                sys_vals = buckets.get("sys", [])
                def_vals = buckets.get("def", [])

                r1_mean = float("nan") if not r1_vals else sum(r1_vals) / len(r1_vals)
                rl_mean = float("nan") if not rl_vals else sum(rl_vals) / len(rl_vals)
                sys_mean = (
                    float("nan") if not sys_vals else sum(sys_vals) / len(sys_vals)
                )
                def_mean = (
                    float("nan") if not def_vals else sum(def_vals) / len(def_vals)
                )

                row = [
                    f"{r:.2f}",
                    f"{r1_mean:.6f}" if not math.isnan(r1_mean) else "",
                    f"{_sem(r1_vals):.6f}" if r1_vals else "",
                    f"{rl_mean:.6f}" if not math.isnan(rl_mean) else "",
                    f"{_sem(rl_vals):.6f}" if rl_vals else "",
                    f"{sys_mean:.6f}" if not math.isnan(sys_mean) else "",
                    f"{_sem(sys_vals):.6f}" if sys_vals else "",
                    f"{def_mean:.6f}" if not math.isnan(def_mean) else "",
                    f"{_sem(def_vals):.6f}" if def_vals else "",
                ]
                w.writerow(row)

        logging.info(f"Wrote CSV: {out_path}")


def plot_csv_sanity(csv_base_dir: Path, plot_base_dir: Path) -> None:
    """
    Read every csv/*/*/*/prompt_*/results.csv and plot sanity errorbar charts:
      - rouge1 vs compression_ratio
      - rougeL vs compression_ratio
      - keep % (system & defense) vs compression_ratio
    """

    results_files = sorted((csv_base_dir / "csv").rglob("results.csv"))
    if not results_files:
        logging.warning(f"No CSVs found under {csv_base_dir / 'csv'}")
        return

    def _read_csv(path: Path):
        xs = []
        rows = []
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)

        def as_float(x: str) -> float:
            if x is None or x == "":
                return float("nan")
            try:
                return float(x)
            except Exception:
                return float("nan")

        tmp = []
        for r in rows:
            tmp.append((as_float(r.get("compression_ratio", "")), r))
        tmp.sort(key=lambda t: t[0])

        xs = [t[0] for t in tmp]
        series = {k: [] for k in (reader.fieldnames or []) if k != "compression_ratio"}
        for _, r in tmp:
            for k in series.keys():
                series[k].append(as_float(r.get(k, "")))
        return xs, series

    for csv_path in results_files:
        # Extract hierarchy: csv/<score_ref>/<defense>/<category>/prompt_X/results.csv
        try:
            # .../csv/<score_ref>/<defense>/<category>/prompt_#/results.csv
            score_ref = csv_path.parents[4].name
            defense = csv_path.parents[3].name
            category = csv_path.parents[2].name
            prompt_dir = csv_path.parents[1].name  # e.g., prompt_0
        except Exception:
            score_ref = "unknown"
            defense = "unknown"
            category = "unknown"
            prompt_dir = "prompt_x"

        out_dir = (
            plot_base_dir / "csv_sanity" / score_ref / defense / category / prompt_dir
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        xs, series = _read_csv(csv_path)

        # ROUGE-1
        import numpy as np

        xs_np = np.array(xs, dtype=float)

        if "rouge1" in series:
            r1 = np.array(series["rouge1"], dtype=float)
            r1e = (
                np.array(series.get("rouge1_err", []), dtype=float)
                if "rouge1_err" in series
                else None
            )
            plt.figure()
            if r1e is not None and len(r1e) == len(r1):
                plt.errorbar(xs_np, r1, yerr=r1e, marker="o", capsize=3)
            else:
                plt.plot(xs_np, r1, marker="o")
            plt.xlabel("Compression ratio")
            plt.ylabel("ROUGE-1 recall")
            plt.ylim(0, 1)
            plt.title(f"ROUGE-1 vs Compression [{score_ref}]")
            plt.grid(True, alpha=0.3)
            plt.savefig(out_dir / "rouge1_csv.png", bbox_inches="tight", dpi=150)
            plt.close()

        # ROUGE-L
        if "rougeL" in series:
            rl = np.array(series["rougeL"], dtype=float)
            rle = (
                np.array(series.get("rougeL_err", []), dtype=float)
                if "rougeL_err" in series
                else None
            )
            plt.figure()
            if rle is not None and len(rle) == len(rl):
                plt.errorbar(xs_np, rl, yerr=rle, marker="o", capsize=3)
            else:
                plt.plot(xs_np, rl, marker="o")
            plt.xlabel("Compression ratio")
            plt.ylabel("ROUGE-L recall")
            plt.ylim(0, 1)
            plt.title(f"ROUGE-L vs Compression [{score_ref}]")
            plt.grid(True, alpha=0.3)
            plt.savefig(out_dir / "rougeL_csv.png", bbox_inches="tight", dpi=150)
            plt.close()

        # Keep % plot (if present)
        sysk = "system_keep_pct"
        defk = "defense_keep_pct"
        have_keep = (sysk in series) or (defk in series)
        if have_keep:
            plt.figure()
            if sysk in series:
                sysv = np.array(series[sysk], dtype=float)
                syse = (
                    np.array(series.get(sysk + "_err", []), dtype=float)
                    if (sysk + "_err") in series
                    else None
                )
                if syse is not None and len(syse) == len(sysv):
                    plt.errorbar(
                        xs_np,
                        sysv,
                        yerr=syse,
                        marker="o",
                        capsize=3,
                        label="System keep %",
                    )
                else:
                    plt.plot(xs_np, sysv, marker="o", label="System keep %")
            if defk in series:
                defv = np.array(series[defk], dtype=float)
                defe = (
                    np.array(series.get(defk + "_err", []), dtype=float)
                    if (defk + "_err") in series
                    else None
                )
                if defe is not None and len(defe) == len(defv):
                    plt.errorbar(
                        xs_np,
                        defv,
                        yerr=defe,
                        marker="o",
                        capsize=3,
                        label="Defense keep %",
                    )
                else:
                    plt.plot(xs_np, defv, marker="o", label="Defense keep %")
            plt.xlabel("Compression ratio")
            plt.ylabel("Keep rate (%)")
            plt.ylim(0, 100)
            plt.title(f"Keep % vs Compression [{score_ref}]")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(out_dir / "keep_pct_csv.png", bbox_inches="tight", dpi=150)
            plt.close()

        logging.info(f"Plotted CSV sanity: {out_dir}")


def main(config: RaccoonPlotterConfig):
    # Create output directories
    run_name_default = config.response_path.parent.name
    run_dir = config.run_name if config.run_name else run_name_default
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

    if config.output_csv:
        save_raccoon_csvs(config.response_path, csv_base_dir=run_path)
        plot_csv_sanity(csv_base_dir=run_path, plot_base_dir=plot_path)
    else:
        plot_results(config.response_path, plot_path)


if __name__ == "__main__":
    main(CLI(RaccoonPlotterConfig, as_positional=False))
