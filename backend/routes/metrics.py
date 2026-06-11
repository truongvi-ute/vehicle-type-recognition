from __future__ import annotations

import csv
import json
from pathlib import Path
from flask import Blueprint, current_app, jsonify, request

metrics_bp = Blueprint("metrics", __name__)


def get_outputs_dir() -> Path:
    return Path(current_app.root_path).parent / "outputs"


@metrics_bp.get("/metrics/runs")
def get_runs() -> tuple[object, int]:
    outputs_dir = get_outputs_dir()
    if not outputs_dir.is_dir():
        return jsonify({"runs": []}), 200

    # Find all relevant JSON and CSV files recursively
    json_files = list(outputs_dir.rglob("*.json"))
    csv_files = list(outputs_dir.rglob("*.csv"))

    runs = []
    # Identify files
    for f in json_files:
        # Ignore data prep config files or generic packages
        if "data_prep" in f.name:
            continue
        rel_path = f.relative_to(outputs_dir)
        runs.append({
            "name": f.stem,
            "filename": f.name,
            "rel_path": str(rel_path).replace("\\", "/"),
            "type": "json"
        })

    for f in csv_files:
        rel_path = f.relative_to(outputs_dir)
        runs.append({
            "name": f.stem,
            "filename": f.name,
            "rel_path": str(rel_path).replace("\\", "/"),
            "type": "csv"
        })

    return jsonify({"runs": sorted(runs, key=lambda x: x["name"])}), 200


@metrics_bp.get("/metrics/curves")
def get_curves() -> tuple[object, int]:
    rel_path = request.args.get("path")
    if not rel_path:
        return jsonify({"error": "Missing 'path' query parameter"}), 400

    outputs_dir = get_outputs_dir()
    file_path = (outputs_dir / rel_path).resolve()

    # Ensure security check so files outside outputs are not read
    if not str(file_path).startswith(str(outputs_dir.resolve())):
        return jsonify({"error": "Access denied"}), 403

    if not file_path.is_file():
        return jsonify({"error": f"File not found: {rel_path}"}), 404

    curves = []
    try:
        if file_path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # If the json is an evaluation dict, look for a nested history or return empty
            if isinstance(data, dict):
                if "history" in data:
                    curves = data["history"]
                else:
                    # Maybe it's just key-value metrics
                    curves = []
            elif isinstance(data, list):
                curves = data
        elif file_path.suffix == ".csv":
            # Parsing YOLO results.csv
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Standardize keys by removing leading/trailing spaces
                    clean_row = {k.strip(): v.strip() for k, v in row.items() if k is not None}
                    epoch = int(clean_row.get("epoch") or 0)
                    
                    # Read Loss
                    train_loss = float(clean_row.get("train/loss") or clean_row.get("train/box_loss") or 0.0)
                    val_loss = float(clean_row.get("val/loss") or clean_row.get("val/box_loss") or 0.0)
                    
                    # Read Accuracy
                    val_acc = float(clean_row.get("metrics/accuracy_top1") or 0.0)
                    
                    curves.append({
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "valid_unseen_loss": val_loss,
                        "valid_unseen_acc": val_acc
                    })
    except Exception as exc:
        return jsonify({"error": f"Failed to parse curves file: {exc}"}), 500

    return jsonify({"curves": curves}), 200


@metrics_bp.get("/metrics/report")
def get_report() -> tuple[object, int]:
    rel_path = request.args.get("path")
    if not rel_path:
        return jsonify({"error": "Missing 'path' query parameter"}), 400

    outputs_dir = get_outputs_dir()
    file_path = (outputs_dir / rel_path).resolve()

    if not str(file_path).startswith(str(outputs_dir.resolve())):
        return jsonify({"error": "Access denied"}), 403

    if not file_path.is_file():
        return jsonify({"error": f"File not found: {rel_path}"}), 404

    if file_path.suffix != ".json":
        return jsonify({"error": "Report must be a JSON file"}), 400

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Pull evaluation report
        test_data = data.get("test", {}) if isinstance(data, dict) else {}
        class_names = data.get("class_names", []) if isinstance(data, dict) else []
        
        # Fallbacks for format
        if not test_data and isinstance(data, dict):
            # Maybe the file itself is a flat classification report
            test_data = data

        return jsonify({
            "accuracy": test_data.get("accuracy"),
            "classification_report": test_data.get("classification_report", {}),
            "class_names": class_names
        }), 200
    except Exception as exc:
        return jsonify({"error": f"Failed to parse report file: {exc}"}), 500


@metrics_bp.get("/metrics/confusion-matrix")
def get_confusion_matrix() -> tuple[object, int]:
    rel_path = request.args.get("path")
    if not rel_path:
        return jsonify({"error": "Missing 'path' query parameter"}), 400

    outputs_dir = get_outputs_dir()
    file_path = (outputs_dir / rel_path).resolve()

    if not str(file_path).startswith(str(outputs_dir.resolve())):
        return jsonify({"error": "Access denied"}), 403

    if not file_path.is_file():
        return jsonify({"error": f"File not found: {rel_path}"}), 404

    if file_path.suffix != ".json":
        return jsonify({"error": "Confusion matrix must be loaded from JSON evaluation files"}), 400

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        test_data = data.get("test", {}) if isinstance(data, dict) else {}
        cm = test_data.get("confusion_matrix", []) if isinstance(test_data, dict) else []
        class_names = data.get("class_names", []) if isinstance(data, dict) else []

        return jsonify({
            "confusion_matrix": cm,
            "class_names": class_names
        }), 200
    except Exception as exc:
        return jsonify({"error": f"Failed to parse confusion matrix: {exc}"}), 500


@metrics_bp.get("/metrics/summary")
def get_summary() -> tuple[object, int]:
    # Compute dynamic summary comparisons across all loaded runs
    outputs_dir = get_outputs_dir()
    if not outputs_dir.is_dir():
        return jsonify({"summary": []}), 200

    evals = list(outputs_dir.rglob("*evaluation*.json")) + list(outputs_dir.rglob("*metrics*.json"))
    summary = []

    for path in evals:
        # Ignore data_prep
        if "data_prep" in path.name:
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extract standard metrics
            test_data = data.get("test", {}) if isinstance(data, dict) else {}
            accuracy = test_data.get("accuracy")
            
            # Pull Macro F1 from classification_report
            report = test_data.get("classification_report", {})
            macro_f1 = report.get("macro avg", {}).get("f1-score")
            weighted_f1 = report.get("weighted avg", {}).get("f1-score")

            # Try to figure out model parameters and inference speed from logs
            # or hardcode standard references if not saved in evaluation
            model_name = path.stem.replace("evaluation_", "").replace("_best", "")
            params = "Unknown"
            speed = "Unknown"
            
            if "resnet50" in model_name.lower():
                params = "24.5 M"
            elif "vit" in model_name.lower():
                params = "86.0 M"
            elif "yolo" in model_name.lower():
                params = "1.45 M"

            summary.append({
                "model": model_name,
                "filename": path.name,
                "rel_path": str(path.relative_to(outputs_dir)).replace("\\", "/"),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "parameters": params,
                "speed": speed
            })
        except Exception:
            continue

    return jsonify({"summary": summary}), 200


@metrics_bp.get("/metrics/dataprep")
def get_dataprep() -> tuple[object, int]:
    outputs_dir = get_outputs_dir()
    file_path = outputs_dir / "data_prep_counts.json"
    if not file_path.is_file():
        return jsonify({"error": "Data prep counts file not found. Run preprocessing first."}), 404
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data), 200
    except Exception as exc:
        return jsonify({"error": f"Failed to read data prep file: {exc}"}), 500


@metrics_bp.get("/metrics/split-metrics")
def get_split_metrics() -> tuple[object, int]:
    rel_path = request.args.get("path")
    if not rel_path:
        return jsonify({"error": "Missing 'path' query parameter"}), 400

    outputs_dir = get_outputs_dir()
    file_path = (outputs_dir / rel_path).resolve()

    if not str(file_path).startswith(str(outputs_dir.resolve())):
        return jsonify({"error": "Access denied"}), 403

    if not file_path.is_file():
        return jsonify({"error": f"File not found: {rel_path}"}), 404

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        splits_comparison = []
        for split_key in ["valid_unseen", "test", "valid_traincopy"]:
            split_data = data.get(split_key)
            if isinstance(split_data, dict):
                report = split_data.get("classification_report", {})
                splits_comparison.append({
                    "split": split_key,
                    "accuracy": split_data.get("accuracy"),
                    "macro_f1": report.get("macro avg", {}).get("f1-score"),
                    "weighted_f1": report.get("weighted avg", {}).get("f1-score")
                })
        return jsonify({"splits": splits_comparison}), 200
    except Exception as exc:
        return jsonify({"error": f"Failed to parse split metrics: {exc}"}), 500


@metrics_bp.get("/metrics/top-errors")
def get_top_errors() -> tuple[object, int]:
    rel_path = request.args.get("path")
    if not rel_path:
        return jsonify({"error": "Missing 'path' query parameter"}), 400

    outputs_dir = get_outputs_dir()
    file_path = (outputs_dir / rel_path).resolve()

    if not str(file_path).startswith(str(outputs_dir.resolve())):
        return jsonify({"error": "Access denied"}), 403

    if not file_path.is_file():
        return jsonify({"error": f"File not found: {rel_path}"}), 404

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        test_data = data.get("test", {}) if isinstance(data, dict) else {}
        cm = test_data.get("confusion_matrix", []) if isinstance(test_data, dict) else []
        class_names = data.get("class_names", []) if isinstance(data, dict) else []

        if not cm or not class_names:
            return jsonify({"errors": []}), 200

        errors = []
        for i, row in enumerate(cm):
            true_class = class_names[i]
            row_total = sum(row)
            for j, val in enumerate(row):
                if i != j and val > 0:
                    pred_class = class_names[j]
                    rate = val / row_total if row_total > 0 else 0.0
                    errors.append({
                        "true_class": true_class,
                        "predicted_class": pred_class,
                        "count": val,
                        "rate": round(rate, 4)
                    })
        
        # Sort by count descending
        errors = sorted(errors, key=lambda x: x["count"], reverse=True)
        return jsonify({"errors": errors[:8]}), 200
    except Exception as exc:
        return jsonify({"error": f"Failed to calculate top errors: {exc}"}), 500

