import logging
from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

from jsonargparse import CLI
from transformers.models.auto.tokenization_auto import AutoTokenizer

from .compressed_context_ifeval import (
    CompressedContextIFEval,
    get_ifeval_results,
    plot_results_across_ratios,
)
from .utils import set_seeds


@dataclass
class HelperCompressedContextIFEvalConfig(CompressedContextIFEval):
    response_dir: Path = Path("must_be_overriden")
    skip_ids: list[str] | None = None
    outputs_base: Path = (
        Path(__file__).resolve().parent / "helper_compressed_context_ifeval"
    )
    seed: int = 42
    output_csv: bool = False


def save_family_csvs(
    ifeval_results_all_ratios: dict[str, dict],
    out_dir: Path,
) -> None:
    """
    Write two CSV files:
      - results_strict.csv
      - results_loose.csv

    Columns:
      compression_ratio, <family_1>, <family_1_err>, <family_2>, <family_2_err>, ..., overall, overall_err
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # Sort ratio keys numerically: "ratio_0.25" -> 0.25
    ratio_keys = sorted(
        ifeval_results_all_ratios.keys(),
        key=lambda k: float(k.split("_", 1)[1]),
    )

    # Union of all families across ratios
    fam_set: set[str] = set()
    for k in ratio_keys:
        byfam = ifeval_results_all_ratios[k].get("by_family") or {}
        fam_set.update(byfam.keys())
    families = sorted(fam_set)

    def write_one(kind: str) -> None:
        # kind is "strict" or "loose"
        out_path = out_dir / f"results_{kind}.csv"
        headers = ["compression_ratio"]
        for fam in families:
            headers.append(fam)
            headers.append(fam + "_err")
        headers += ["overall", "overall_err"]

        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(headers)

            for k in ratio_keys:
                ratio = float(k.split("_", 1)[1])
                res = ifeval_results_all_ratios[k]
                byfam = res.get("by_family") or {}
                scores = res.get("scores") or {}

                overall = scores.get(f"instruction_{kind}", "")
                overall_err = scores.get(f"overall_err_{kind}", "")

                row = [f"{ratio:.2f}"]
                for fam in families:
                    rec = byfam.get(fam) or {}
                    val = rec.get(kind, "")
                    err = rec.get(f"{kind}_err", "")
                    row.append(f"{val:.6f}" if isinstance(val, (int, float)) else "")
                    row.append(f"{err:.6f}" if isinstance(err, (int, float)) else "")
                row.append(
                    f"{overall:.6f}" if isinstance(overall, (int, float)) else ""
                )
                row.append(
                    f"{overall_err:.6f}"
                    if isinstance(overall_err, (int, float))
                    else ""
                )
                w.writerow(row)

    write_one("strict")
    write_one("loose")


def plot_csv_sanity(csv_dir: Path, plot_dir: Path) -> None:
    """
    Read results_{strict,loose}.csv and plot:
      - overall (with error bars if available) vs compression_ratio
      - per-family curves (with error bars if available) vs compression_ratio
    Saves into plot_dir / 'csv_sanity'.
    """
    plot_out = plot_dir / "csv_sanity"
    plot_out.mkdir(parents=True, exist_ok=True)

    def _read_csv(path: Path) -> tuple[list[float], dict[str, list[float]]]:
        """
        Returns:
          xs: sorted compression ratios
          series: column -> list of floats (NaN if blank)
        """
        xs: list[float] = []
        rows: list[dict[str, str]] = []
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            for r in reader:
                rows.append(r)

        def as_float(x: str) -> float:
            if x is None or x == "":
                return float("nan")
            try:
                return float(x)
            except Exception:
                return float("nan")

        # Collect and sort by compression_ratio
        tmp = []
        for r in rows:
            ratio = as_float(r.get("compression_ratio", ""))
            tmp.append((ratio, r))
        tmp.sort(key=lambda t: t[0])

        # Build series
        keys = [h for h in (headers or []) if h != "compression_ratio"]
        series: dict[str, list[float]] = {h: [] for h in keys}
        for ratio, r in tmp:
            xs.append(ratio)
            for h in keys:
                series[h].append(as_float(r.get(h, "")))

        return xs, series

    def _plot_one(csv_path: Path, kind: str) -> None:
        xs, series = _read_csv(csv_path)
        xs_np = np.array(xs, dtype=float)

        # Overall
        overall = np.array(series.get("overall", []), dtype=float)
        overall_err = np.array(series.get("overall_err", []), dtype=float)
        plt.figure()
        if "overall_err" in series:
            plt.errorbar(xs_np, overall, yerr=overall_err, marker="o", capsize=3)
        else:
            plt.plot(xs_np, overall, marker="o")
        plt.xlabel("Compression ratio")
        plt.ylabel("Overall")
        plt.title(f"Overall ({kind}) vs Compression Ratio [CSV]")
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_out / f"overall_{kind}_csv.png", bbox_inches="tight")
        plt.close()

        # Families (everything except overall/overall_err)
        fam_keys = [
            k
            for k in series.keys()
            if k not in ("overall", "overall_err") and not k.endswith("_err")
        ]

        # All families in one figure
        plt.figure()
        for fam in sorted(fam_keys):
            ys = np.array(series[fam], dtype=float)
            if np.all(np.isnan(ys)):
                continue
            err_key = fam + "_err"
            if err_key in series:
                errs = np.array(series[err_key], dtype=float)
                plt.errorbar(xs_np, ys, yerr=errs, marker="o", capsize=3, label=fam)
            else:
                plt.plot(xs_np, ys, marker="o", label=fam)
        plt.xlabel("Compression ratio")
        plt.ylabel(kind.capitalize())
        plt.title(f"All Families ({kind}) vs Compression Ratio [CSV]")
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)
        if fam_keys:
            plt.legend(
                title="Family",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                borderaxespad=0.0,
            )
        plt.savefig(plot_out / f"families_all_{kind}_csv.png", bbox_inches="tight")
        plt.close()

        # Per-family individual plots (optional but useful)
        fam_dir = plot_out / f"families_{kind}"
        fam_dir.mkdir(parents=True, exist_ok=True)
        for fam in sorted(fam_keys):
            ys = np.array(series[fam], dtype=float)
            if np.all(np.isnan(ys)):
                continue
            plt.figure()
            err_key = fam + "_err"
            if err_key in series:
                errs = np.array(series[err_key], dtype=float)
                plt.errorbar(xs_np, ys, yerr=errs, marker="o", capsize=3)
            else:
                plt.plot(xs_np, ys, marker="o")
            plt.xlabel("Compression ratio")
            plt.ylabel(kind.capitalize())
            plt.title(f"{fam} ({kind}) vs Compression Ratio [CSV]")
            plt.ylim(0.0, 1.0)
            plt.grid(True, alpha=0.3)
            safe = fam.replace(":", "_").replace("/", "_")
            plt.savefig(fam_dir / f"{safe}_{kind}_csv.png", bbox_inches="tight")
            plt.close()

    strict_csv = csv_dir / "results_strict.csv"
    loose_csv = csv_dir / "results_loose.csv"

    if strict_csv.exists():
        _plot_one(strict_csv, "strict")
    if loose_csv.exists():
        _plot_one(loose_csv, "loose")


def save_single_multi_csvs(
    ifeval_results_all_ratios: dict[str, dict], out_dir: Path
) -> None:
    """
    Write four CSVs:
      - single_results_strict.csv / single_results_loose.csv
      - multi_results_strict.csv  / multi_results_loose.csv

    Columns:
      compression_ratio, <family_1>, <family_1_err>, ..., overall, overall_err
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    ratio_keys = sorted(
        ifeval_results_all_ratios.keys(), key=lambda k: float(k.split("_", 1)[1])
    )

    # collect family sets
    fams_single, fams_multi = set(), set()
    for k in ratio_keys:
        res = ifeval_results_all_ratios[k]
        fams_single |= set((res.get("by_family_single") or {}).keys())
        fams_multi |= set((res.get("by_family_multi") or {}).keys())
    fams_single = sorted(fams_single)
    fams_multi = sorted(fams_multi)

    def _write(kind: str, which: str) -> None:
        # kind: "strict" | "loose"; which: "single" | "multi"
        fams = fams_single if which == "single" else fams_multi
        out_path = out_dir / f"{which}_results_{kind}.csv"
        headers = (
            ["compression_ratio"]
            + [x for f in fams for x in (f, f + "_err")]
            + ["overall", "overall_err"]
        )

        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(headers)
            for k in ratio_keys:
                ratio = float(k.split("_", 1)[1])
                res = ifeval_results_all_ratios[k]
                byfam = res.get(f"by_family_{which}") or {}
                sm = res.get("single_multi", {}).get(which, {})  # overall & errs

                overall = sm.get(f"instruction_{kind}", "")
                overall_err = sm.get(f"instruction_err_{kind}", "")

                row = [f"{ratio:.2f}"]
                for fam in fams:
                    rec = byfam.get(fam) or {}
                    val = rec.get(kind, "")
                    err = rec.get(f"{kind}_err", "")
                    row += [
                        f"{val:.6f}" if isinstance(val, (int, float)) else "",
                        f"{err:.6f}" if isinstance(err, (int, float)) else "",
                    ]
                row += [
                    f"{overall:.6f}" if isinstance(overall, (int, float)) else "",
                    f"{overall_err:.6f}"
                    if isinstance(overall_err, (int, float))
                    else "",
                ]
                w.writerow(row)

    for which in ("single", "multi"):
        _write("strict", which)
        _write("loose", which)


def plot_single_multi_csvs(csv_dir: Path, plot_dir: Path) -> None:
    """
    Quick sanity plots from the CSVs above into plot_dir / 'csv_sanity_single_multi'.
    """
    out = plot_dir / "csv_sanity_single_multi"
    out.mkdir(parents=True, exist_ok=True)

    def _read(path: Path):
        xs, rows = [], []
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            hdrs = reader.fieldnames or []
            for r in reader:
                rows.append(r)

        def asf(x):
            try:
                return float(x)
            except Exception:
                return float("nan")

        tmp = [(asf(r.get("compression_ratio", "")), r) for r in rows]
        tmp.sort(key=lambda t: t[0])
        xs = [t[0] for t in tmp]
        series = {
            h: [asf(r.get(h, "")) for _, r in tmp]
            for h in hdrs
            if h != "compression_ratio"
        }
        return np.array(xs, dtype=float), series

    for which in ("single", "multi"):
        for kind in ("strict", "loose"):
            p = csv_dir / f"{which}_results_{kind}.csv"
            if not p.exists():
                continue
            xs, series = _read(p)

            # overall
            y = np.array(series.get("overall", []), dtype=float)
            e = np.array(series.get("overall_err", []), dtype=float)
            plt.figure()
            if "overall_err" in series:
                plt.errorbar(xs, y, yerr=e, marker="o", capsize=3)
            else:
                plt.plot(xs, y, marker="o")
            plt.xlabel("Compression ratio")
            plt.ylabel("Overall")
            plt.title(f"{which.title()} — Overall ({kind}) [CSV]")
            plt.ylim(0.0, 1.0)
            plt.grid(True, alpha=0.3)
            plt.savefig(out / f"{which}_overall_{kind}.png", bbox_inches="tight")
            plt.close()

            # all families overlay
            fam_keys = [
                k
                for k in series.keys()
                if k not in ("overall", "overall_err") and not k.endswith("_err")
            ]
            plt.figure()
            for fam in sorted(fam_keys):
                ys = np.array(series[fam], dtype=float)
                if np.all(np.isnan(ys)):
                    continue
                errk = fam + "_err"
                if errk in series:
                    es = np.array(series[errk], dtype=float)
                    plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=fam)
                else:
                    plt.plot(xs, ys, marker="o", label=fam)
            plt.xlabel("Compression ratio")
            plt.ylabel(kind.capitalize())
            plt.title(f"{which.title()} — All Families ({kind}) [CSV]")
            plt.ylim(0.0, 1.0)
            plt.grid(True, alpha=0.3)
            if fam_keys:
                plt.legend(
                    title="Family",
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left",
                    borderaxespad=0.0,
                )
            plt.savefig(out / f"{which}_families_all_{kind}.png", bbox_inches="tight")
            plt.close()


def main(config: HelperCompressedContextIFEvalConfig):
    set_seeds(config.seed)

    # Create output directories
    run_name_default = config.response_dir.parent.name
    if config.output_csv:
        run_name_default += "_csv"
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

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        cache_dir=config.model_cache_dir,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    ratio_dirs = sorted(config.response_dir.glob("ratio_*"))

    ifeval_results_all_ratios = {}
    for ratio_dir in ratio_dirs:
        response_path = ratio_dir / "responses.jsonl"
        results = get_ifeval_results(
            input_path=config.sys_ifeval_path,
            response_path=response_path,
            skip_ids=config.skip_ids,
            tokenizer=tokenizer,
        )
        ifeval_results_all_ratios[ratio_dir.name] = results

    if config.output_csv:
        save_family_csvs(ifeval_results_all_ratios, out_dir=run_path)
        save_single_multi_csvs(ifeval_results_all_ratios, out_dir=run_path)
        plot_csv_sanity(csv_dir=run_path, plot_dir=plot_path)
        plot_single_multi_csvs(csv_dir=run_path, plot_dir=plot_path)
    else:
        plot_results_across_ratios(ifeval_results_all_ratios, plot_path=plot_path)


if __name__ == "__main__":
    main(CLI(HelperCompressedContextIFEvalConfig, as_positional=False))
