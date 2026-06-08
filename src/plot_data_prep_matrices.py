"""
Plot 2D matrix views from the data_prep JSON report.

Input:
    outputs/data_prep_counts.json

Output:
    outputs/figures/data_prep_<matrix_name>.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_MATRICES = [
    "split_dataset_matrix",
    "augmentation_summary_matrix",
]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON report: {path}")
    return payload


def require_plot_packages():
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ModuleNotFoundError as exc:
        missing = exc.name or "plotting dependency"
        raise SystemExit(
            f"Missing Python package '{missing}'. Install project requirements first, "
            "for example: pip install -r requirements.txt"
        ) from exc
    return plt, pd, sns


def matrix_to_dataframe(matrix: Dict[str, Any]):
    columns = matrix.get("columns")
    rows = matrix.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        raise ValueError("Matrix must contain 'columns' and 'rows' lists.")
    _, pd, _ = require_plot_packages()
    return pd.DataFrame(rows, columns=[str(column) for column in columns])


def numeric_heatmap_frame(df):
    _, pd, _ = require_plot_packages()
    if "class" in df.columns:
        df = df.set_index("class")

    converted = df.copy()
    for column in converted.columns:
        if converted[column].dtype == bool:
            converted[column] = converted[column].astype(int)
        else:
            converted[column] = pd.to_numeric(converted[column], errors="coerce")

    converted = converted.dropna(axis=1, how="all")
    if converted.empty:
        raise ValueError("Matrix does not contain numeric columns to plot.")
    return converted.fillna(0)


def heatmap_scale_bounds(values):
    if "TOTAL" in values.index:
        scaled_values = values.drop(index="TOTAL")
    else:
        scaled_values = values
    if scaled_values.empty:
        scaled_values = values
    vmin = float(scaled_values.min().min())
    vmax = float(scaled_values.max().max())
    if vmin == vmax:
        vmax = vmin + 1.0
    return vmin, vmax


def draw_total_row(ax, plt, values) -> None:
    if "TOTAL" not in values.index:
        return

    row_index = list(values.index).index("TOTAL")
    for col_index, value in enumerate(values.loc["TOTAL"]):
        ax.add_patch(
            plt.Rectangle(
                (col_index, row_index),
                1,
                1,
                facecolor="#f4f4f5",
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        )
        ax.text(
            col_index + 0.5,
            row_index + 0.5,
            f"{float(value):.0f}",
            ha="center",
            va="center",
            color="#111827",
            fontsize=9,
            zorder=4,
        )


def plot_matrix(matrix: Dict[str, Any], title: str, output_path: Path) -> None:
    plt, _, sns = require_plot_packages()
    df = matrix_to_dataframe(matrix)
    values = numeric_heatmap_frame(df)
    heat_values = values.copy()
    if "TOTAL" in heat_values.index:
        heat_values.loc["TOTAL"] = float("nan")
    vmin, vmax = heatmap_scale_bounds(values)

    width = max(8.0, values.shape[1] * 1.35 + 3.0)
    height = max(4.8, values.shape[0] * 0.52 + 2.2)

    sns.set_theme(style="white", font_scale=0.9)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        heat_values,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        cbar=True,
        ax=ax,
    )
    draw_total_row(ax, plt, values)
    ax.set_title(title, pad=14, fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("class")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_all(report_path: Path, out_dir: Path, matrices: List[str]) -> List[Path]:
    report = read_json(report_path)
    matrix_root = report.get("matrices")
    if not isinstance(matrix_root, dict):
        raise ValueError(f"Report does not contain a 'matrices' object: {report_path}")

    saved: List[Path] = []
    for matrix_name in matrices:
        matrix = matrix_root.get(matrix_name)
        if not isinstance(matrix, dict):
            print(f"Skipped missing matrix: {matrix_name}")
            continue

        output_path = out_dir / f"data_prep_{matrix_name}.png"
        plot_matrix(matrix, matrix_name.replace("_", " ").title(), output_path)
        saved.append(output_path)
        print(f"Saved matrix plot: {output_path}")
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot data_prep matrix heatmaps with seaborn.")
    parser.add_argument("--report", type=Path, default=Path("outputs/data_prep_counts.json"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument(
        "--matrix",
        action="append",
        choices=DEFAULT_MATRICES,
        help="Matrix name to plot. Repeat to plot multiple. Defaults to all matrices.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrices = args.matrix or DEFAULT_MATRICES
    saved = plot_all(args.report, args.out_dir, matrices)
    if not saved:
        raise SystemExit("No matrix plots were generated.")


if __name__ == "__main__":
    main()
