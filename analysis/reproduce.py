"""Recompute released study outcomes. No private files or network access required."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qcnn_study.reproduction import analyze, load_inputs, render_tables


def plot(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    output = ROOT / "figures"
    output.mkdir(exist_ok=True)
    cases = result["benchmark_reference"]
    fig, ax = plt.subplots(figsize=(11.8, 6.1))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.27, top=0.77)
    fig.text(0.08, 0.95, "QCNN benchmark: cross-study comparison", fontsize=19, weight="bold")
    fig.text(0.08, 0.89, "50 images per case | Unpaired literature reference | Heterogeneous post-processing", color="#475569")
    series = [
        ("LQCloud raw (exploratory)", [c["lqcloud_raw"]["correct"] / c["lqcloud_raw"]["total"] for c in cases], "#167d9a", -0.25),
        ("IonQ Aria 1 + EM (paper, approx.)", [c["paper_aria_em_accuracy_approx"] for c in cases], "#e0783d", 0),
        ("Ideal simulation (paper, approx.)", [c["paper_ideal_accuracy_approx"] for c in cases], "#a9b5c5", 0.25),
    ]
    for label, values, color, offset in series:
        bars = ax.bar([i+offset for i in range(len(cases))], [100*v for v in values], 0.23, color=color, label=label)
        for bar, value in zip(bars, values):
            prefix = "~" if offset != -0.25 else ""
            ax.text(bar.get_x()+bar.get_width()/2, 100*value+1.5, f"{prefix}{100*value:.0f}%", ha="center", fontsize=10)
    ax.set_xticks(range(len(cases)), [c["case"] + ("\nBalanced local labels" if c["local_asset_label_counts"] == [25,25] else "\nSingle-class local labels") for c in cases])
    ax.set_ylim(0, 115)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Classification accuracy (%)")
    ax.yaxis.grid(True, alpha=0.18)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.15), ncol=3, frameon=False, fontsize=9)
    fig.text(0.08, 0.105, "Literature: IonQ white paper, Section 2.5 / Figure 8. Approximate values digitized from the figure.", fontsize=10, color="#475569")
    fig.text(0.08, 0.055, "Exploratory selection and class imbalance limit interpretation. Hardware equivalence is not established.", fontsize=10, color="#475569")
    fig.savefig(output / "benchmark_reference.png", dpi=170, metadata={"Software": "QCNN study analysis"})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.8, 5.8))
    fig.subplots_adjust(left=0.35, right=0.97, bottom=0.31, top=0.73)
    fig.text(0.06, 0.94, "QCNN evaluation: independent 200-image holdout", fontsize=18, weight="bold")
    fig.text(0.06, 0.875, "20 separate calibration images | Balanced test set | No matched IonQ observations", color="#475569")
    entries = [("QPU / original head", "qpu_original_head", "#8893a3", False),
               ("QPU / supervised calibration", "qpu_calibrated", "#167d9a", True),
               ("Exact noiseless / calibration", "ideal_calibrated", "#576ca8", True)]
    for i, (_, key, color, interval) in enumerate(entries):
        row = result["holdout"][key]
        value = 100*row["accuracy"]
        if interval:
            low, high = [100*x for x in row["wilson_95"]]
            ax.errorbar(value, i, xerr=[[value-low], [high-value]], fmt="o", capsize=5, color=color, markersize=9, linewidth=2)
            note = f"{value:.0f}%   [{low:.2f}, {high:.2f}]"
        else:
            ax.plot(value, i, "o", color=color, markersize=9)
            note = "50%   (single-class predictions)"
        ax.text(value, i-0.2, note, ha="center", va="bottom", color=color)
    ax.set_yticks(range(3), [entry[0] for entry in entries])
    ax.set_ylim(2.55, -0.65)
    ax.set_xlim(0, 104)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Classification accuracy (%)")
    ax.xaxis.grid(True, alpha=0.18)
    ax.set_axisbelow(True)
    fig.text(0.06, 0.14, "Error bars: conditional image-level Wilson 95% intervals; calibration and temporal uncertainty are excluded.", fontsize=10, color="#475569")
    fig.text(0.06, 0.085, "The ideal and QPU decision layers are fitted separately; the contrast is not a noise-only ablation.", fontsize=10, color="#475569")
    fig.text(0.06, 0.035, "Cross-study accuracy differences do not identify comparative hardware fidelity.", fontsize=10, color="#475569")
    fig.savefig(output / "holdout.png", dpi=170, metadata={"Software": "QCNN study analysis"})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare regenerated statistics to committed results without writing")
    parser.add_argument("--plot", action="store_true", help="Regenerate figures; requires matplotlib")
    args = parser.parse_args()
    if args.check and args.plot:
        parser.error("--check is read-only and cannot be combined with --plot")
    result = analyze(load_inputs(ROOT / "data"))
    outputs = {
        "statistics.json": json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        "tables.md": render_tables(result),
    }
    target = ROOT / "results"
    if args.check:
        for name, content in outputs.items():
            if (target / name).read_text(encoding="utf-8") != content:
                raise SystemExit(f"Published result differs from recalculation: {name}")
        print("Verified: all released holdout, paired, and temporal statistics match the published outputs.")
    else:
        target.mkdir(exist_ok=True)
        for name, content in outputs.items():
            (target / name).write_text(content, encoding="utf-8")
        print("Recomputed results/statistics.json and results/tables.md from released data.")
    if args.plot:
        plot(result)
        print("Rebuilt both figures from the same calculated results.")


if __name__ == "__main__":
    main()
