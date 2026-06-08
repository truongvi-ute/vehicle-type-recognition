"""
Create report-ready SVG charts from training/evaluation JSON files.

This script intentionally uses only the Python standard library so it can run
even when matplotlib/seaborn are not installed.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


CLASS_NAMES = [
    "bicycle",
    "boat",
    "bus",
    "car",
    "helicopter",
    "minibus",
    "motorcycle",
    "taxi",
    "train",
    "truck",
]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_svg(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _scale(value: float, old_min: float, old_max: float, new_min: float, new_max: float) -> float:
    if old_max == old_min:
        return (new_min + new_max) / 2
    return new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min)


def _polyline(points: Iterable[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def plot_loss(history_path: Path, output_path: Path) -> None:
    rows = _read_json(history_path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"History file is empty or invalid: {history_path}")

    epochs = [float(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    valid_loss = [float(row["valid_unseen_loss"]) for row in rows]

    width, height = 980, 580
    left, right, top, bottom = 88, 40, 56, 82
    plot_w = width - left - right
    plot_h = height - top - bottom

    min_epoch, max_epoch = min(epochs), max(epochs)
    min_loss = min(train_loss + valid_loss)
    max_loss = max(train_loss + valid_loss)
    pad = max((max_loss - min_loss) * 0.08, 0.03)
    y_min = max(0.0, min_loss - pad)
    y_max = max_loss + pad

    def x_pos(epoch: float) -> float:
        return _scale(epoch, min_epoch, max_epoch, left, left + plot_w)

    def y_pos(loss: float) -> float:
        return _scale(loss, y_min, y_max, top + plot_h, top)

    train_points = [(x_pos(e), y_pos(v)) for e, v in zip(epochs, train_loss)]
    valid_points = [(x_pos(e), y_pos(v)) for e, v in zip(epochs, valid_loss)]

    y_ticks = 6
    grid_lines = []
    labels = []
    for i in range(y_ticks + 1):
        value = y_min + (y_max - y_min) * i / y_ticks
        y = y_pos(value)
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" '
            'stroke="#e7e8ec" stroke-width="1"/>'
        )
        labels.append(
            f'<text x="{left - 14}" y="{y + 5:.2f}" text-anchor="end" '
            'font-size="13" fill="#596172">{value:.2f}</text>'
        )

    x_labels = []
    for epoch in epochs:
        if int(epoch) == epoch and (int(epoch) == 1 or int(epoch) % 2 == 0 or epoch == max_epoch):
            x = x_pos(epoch)
            x_labels.append(
                f'<text x="{x:.2f}" y="{height - 42}" text-anchor="middle" '
                'font-size="12" fill="#596172">{int(epoch)}</text>'
            )

    best_idx = min(range(len(valid_loss)), key=valid_loss.__getitem__)
    best_epoch = int(epochs[best_idx])
    best_loss = valid_loss[best_idx]
    best_x = x_pos(epochs[best_idx])
    best_y = y_pos(best_loss)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{left}" y="34" font-size="24" font-weight="700" fill="#111827">ResNet-50 Loss Curves</text>
  <text x="{left}" y="55" font-size="13" fill="#596172">Training loss and valid_unseen loss by epoch</text>
  {"".join(grid_lines)}
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#2d3748" stroke-width="1.4"/>
  <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#2d3748" stroke-width="1.4"/>
  {"".join(labels)}
  {"".join(x_labels)}
  <text x="{left + plot_w / 2:.2f}" y="{height - 14}" text-anchor="middle" font-size="14" fill="#374151">Epoch</text>
  <text transform="translate(24 {top + plot_h / 2:.2f}) rotate(-90)" text-anchor="middle" font-size="14" fill="#374151">Loss</text>
  <polyline fill="none" stroke="#2563eb" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" points="{_polyline(train_points)}"/>
  <polyline fill="none" stroke="#dc2626" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" points="{_polyline(valid_points)}"/>
  <circle cx="{best_x:.2f}" cy="{best_y:.2f}" r="5" fill="#dc2626" stroke="#ffffff" stroke-width="2"/>
  <line x1="{best_x:.2f}" y1="{top}" x2="{best_x:.2f}" y2="{top + plot_h}" stroke="#dc2626" stroke-width="1.2" stroke-dasharray="5 5" opacity="0.7"/>
  <rect x="{width - 276}" y="31" width="226" height="82" rx="8" fill="#f8fafc" stroke="#e2e8f0"/>
  <line x1="{width - 252}" y1="58" x2="{width - 212}" y2="58" stroke="#2563eb" stroke-width="3"/>
  <text x="{width - 200}" y="63" font-size="14" fill="#111827">train_loss</text>
  <line x1="{width - 252}" y1="86" x2="{width - 212}" y2="86" stroke="#dc2626" stroke-width="3"/>
  <text x="{width - 200}" y="91" font-size="14" fill="#111827">valid_unseen_loss</text>
  <text x="{left + 14}" y="{top + 24}" font-size="13" fill="#991b1b">Best valid: epoch {best_epoch}, loss {best_loss:.4f}</text>
</svg>
"""
    _write_svg(output_path, svg)


def _find_confusion_matrix(evaluation: dict[str, Any], split: str) -> tuple[list[str], list[list[int]]]:
    split_metrics = evaluation.get(split)
    if not isinstance(split_metrics, dict):
        raise ValueError(f"Split '{split}' not found in evaluation JSON.")

    matrix = split_metrics.get("confusion_matrix")
    if not matrix:
        raise ValueError(
            f"Split '{split}' does not contain confusion_matrix. "
            "Run src/evaluate.py first with the trained checkpoint."
        )

    class_names = evaluation.get("class_names") or CLASS_NAMES
    return [str(name) for name in class_names], [[int(value) for value in row] for row in matrix]


def plot_confusion_matrix(evaluation_path: Path, output_path: Path, split: str = "test") -> None:
    evaluation = _read_json(evaluation_path)
    if not isinstance(evaluation, dict):
        raise ValueError(f"Evaluation file is invalid: {evaluation_path}")

    class_names, matrix = _find_confusion_matrix(evaluation, split)
    n = len(class_names)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError("Confusion matrix shape does not match class_names.")

    width, height = 980, 880
    left, top = 190, 98
    cell = 62
    max_value = max(max(row) for row in matrix) or 1

    cells = []
    for r, row in enumerate(matrix):
        row_total = sum(row) or 1
        for c, value in enumerate(row):
            intensity = value / max_value
            blue = int(244 - 160 * intensity)
            green = int(248 - 98 * intensity)
            red = int(239 - 212 * intensity)
            fill = f"rgb({red},{green},{blue})"
            text_fill = "#ffffff" if intensity > 0.55 else "#111827"
            pct = value / row_total * 100
            x = left + c * cell
            y = top + r * cell
            cells.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#ffffff" stroke-width="1"/>'
                f'<text x="{x + cell / 2}" y="{y + 26}" text-anchor="middle" font-size="16" font-weight="700" fill="{text_fill}">{value}</text>'
                f'<text x="{x + cell / 2}" y="{y + 45}" text-anchor="middle" font-size="11" fill="{text_fill}">{pct:.1f}%</text>'
            )

    row_labels = []
    col_labels = []
    for i, name in enumerate(class_names):
        escaped = html.escape(name)
        y = top + i * cell + cell / 2 + 5
        x = left + i * cell + cell / 2
        row_labels.append(
            f'<text x="{left - 14}" y="{y:.2f}" text-anchor="end" font-size="13" fill="#374151">{escaped}</text>'
        )
        col_labels.append(
            f'<text transform="translate({x:.2f} {top - 14}) rotate(-45)" text-anchor="start" font-size="13" fill="#374151">{escaped}</text>'
        )

    accuracy = evaluation.get(split, {}).get("accuracy")
    accuracy_text = f"Accuracy: {float(accuracy) * 100:.2f}%" if accuracy is not None else ""

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="46" y="38" font-size="24" font-weight="700" fill="#111827">ResNet-50 Confusion Matrix</text>
  <text x="46" y="62" font-size="13" fill="#596172">Split: {html.escape(split)}. Rows are true labels, columns are predicted labels. {accuracy_text}</text>
  {"".join(col_labels)}
  {"".join(row_labels)}
  {"".join(cells)}
  <text x="{left + n * cell / 2:.2f}" y="{top + n * cell + 58}" text-anchor="middle" font-size="15" fill="#374151">Predicted label</text>
  <text transform="translate(44 {top + n * cell / 2:.2f}) rotate(-90)" text-anchor="middle" font-size="15" fill="#374151">True label</text>
</svg>
"""
    _write_svg(output_path, svg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw loss curve and confusion matrix SVG files.")
    parser.add_argument("--history", type=Path, default=Path("outputs/history_resnet50.json"))
    parser.add_argument("--evaluation", type=Path, default=Path("outputs/evaluation_resnet50_best.json"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loss_path = args.out_dir / "resnet50_loss.svg"
    plot_loss(args.history, loss_path)
    print(f"Saved loss plot: {loss_path}")

    if args.evaluation.is_file():
        cm_path = args.out_dir / f"resnet50_confusion_matrix_{args.split}.svg"
        plot_confusion_matrix(args.evaluation, cm_path, split=args.split)
        print(f"Saved confusion matrix: {cm_path}")
    else:
        print(
            f"Skipped confusion matrix: {args.evaluation} does not exist. "
            "Generate it with src/evaluate.py first."
        )


if __name__ == "__main__":
    main()
