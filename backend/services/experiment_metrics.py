from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class ExperimentDataError(ValueError):
    pass


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise ExperimentDataError(f"Missing data file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentDataError(f"Cannot read JSON file {path}: {exc}") from exc


def _safe_output_path(outputs_dir: Path, relative_path: str) -> Path:
    path = (outputs_dir / relative_path).resolve()
    root = outputs_dir.resolve()
    if path != root and root not in path.parents:
        raise ExperimentDataError(f"Output path escapes project directory: {relative_path}")
    return path


def _as_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentDataError(f"{label} must be numeric, found {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ExperimentDataError(f"{label} must be finite, found {value!r}")
    return result


def _as_optional_number(value: Any) -> float | None:
    if value is None:
        return None
    return _as_number(value, "optional metric")


def _as_count(value: Any, label: str) -> int:
    number = _as_number(value, label)
    if number < 0 or not number.is_integer():
        raise ExperimentDataError(f"{label} must be a non-negative integer, found {value!r}")
    return int(number)


def _registry(outputs_dir: Path) -> dict[str, Any]:
    registry = _read_json(outputs_dir / "experiment_registry.json")
    if not isinstance(registry, dict) or not isinstance(registry.get("experiments"), list):
        raise ExperimentDataError("experiment_registry.json has an invalid structure")
    return registry


def _lookup(items: list[dict[str, Any]], item_id: str, label: str) -> dict[str, Any]:
    for item in items:
        if item.get("id") == item_id:
            return item
    raise ExperimentDataError(f"Unknown {label}: {item_id}")


def _experiment_files_exist(outputs_dir: Path, experiment: dict[str, Any]) -> bool:
    required = [experiment.get("evaluation"), experiment.get("history")]
    return all(
        isinstance(relative_path, str)
        and _safe_output_path(outputs_dir, relative_path).is_file()
        for relative_path in required
    )


def build_catalog(outputs_dir: Path) -> dict[str, Any]:
    registry = _registry(outputs_dir)
    experiments = registry["experiments"]
    by_key = {
        (item["model"], item["dataset"], item["version"]): item
        for item in experiments
    }

    combinations = []
    for model in registry.get("models", []):
        for dataset in registry.get("datasets", []):
            for version in registry.get("versions", []):
                key = (model["id"], dataset["id"], version["id"])
                experiment = by_key.get(key)
                available = bool(
                    experiment and _experiment_files_exist(outputs_dir, experiment)
                )
                combinations.append(
                    {
                        "id": experiment.get("id") if experiment else "__".join(key),
                        "model": model["id"],
                        "dataset": dataset["id"],
                        "version": version["id"],
                        "available": available,
                        "status": "ready" if available else "missing",
                    }
                )

    return {
        "models": registry.get("models", []),
        "datasets": registry.get("datasets", []),
        "versions": registry.get("versions", []),
        "combinations": combinations,
    }


def _dataset_profile_entry(
    registry: dict[str, Any], dataset_id: str, version_id: str
) -> dict[str, Any]:
    for profile in registry.get("dataset_profiles", []):
        if profile.get("dataset") == dataset_id and profile.get("version") == version_id:
            return profile
    raise ExperimentDataError(
        f"Missing dataset profile for dataset={dataset_id}, version={version_id}"
    )


def _normalize_base_dataprep(data: dict[str, Any]) -> dict[str, Any]:
    snapshot = data.get("snapshot")
    if not isinstance(snapshot, dict):
        raise ExperimentDataError("Base data-prep report is missing snapshot")

    raw = snapshot.get("raw", {})
    splits = snapshot.get("splits", {})
    per_split = splits.get("per_split", {}) if isinstance(splits, dict) else {}
    totals = splits.get("totals", {}) if isinstance(splits, dict) else {}
    if not isinstance(raw, dict) or not isinstance(per_split, dict):
        raise ExperimentDataError("Base data-prep report has invalid raw/split data")

    raw_per_class = raw.get("per_class")
    if not isinstance(raw_per_class, dict) or not raw_per_class:
        raise ExperimentDataError("Base data-prep report is missing raw.per_class")
    class_names = sorted(raw_per_class)
    distribution = []
    for class_name in class_names:
        for split_name in ["train", "valid_unseen", "valid_traincopy", "test"]:
            if not isinstance(per_split.get(split_name), dict):
                raise ExperimentDataError(f"Base data-prep report is missing split {split_name}")
        distribution.append(
            {
                "class_name": class_name,
                "raw": _as_count(raw_per_class.get(class_name), f"raw.{class_name}"),
                "train": _as_count(
                    per_split["train"].get(class_name), f"train.{class_name}"
                ),
                "valid_unseen": _as_count(
                    per_split["valid_unseen"].get(class_name),
                    f"valid_unseen.{class_name}",
                ),
                "valid_traincopy": _as_count(
                    per_split["valid_traincopy"].get(class_name),
                    f"valid_traincopy.{class_name}",
                ),
                "test": _as_count(
                    per_split["test"].get(class_name), f"test.{class_name}"
                ),
            }
        )

    raw_total = _as_count(raw.get("total"), "raw.total")
    split_totals = {
        split_name: _as_count(totals.get(split_name), f"splits.totals.{split_name}")
        for split_name in ["train", "valid_unseen", "valid_traincopy", "test"]
    }
    if raw_total != sum(row["raw"] for row in distribution):
        raise ExperimentDataError("raw.total does not match the per-class raw counts")
    for split_name, declared_total in split_totals.items():
        if declared_total != sum(row[split_name] for row in distribution):
            raise ExperimentDataError(
                f"splits.totals.{split_name} does not match the per-class counts"
            )

    return {
        "raw_total": raw_total,
        "split_totals": split_totals,
        "class_distribution": distribution,
    }


def _normalize_augmentation(data: dict[str, Any]) -> dict[str, Any]:
    snapshot = data.get("snapshot")
    if isinstance(snapshot, dict):
        augmented = snapshot.get("augmented", {})
        rows = augmented.get("train_per_class_bucket", {}) if isinstance(augmented, dict) else {}
        if not isinstance(rows, dict):
            raise ExperimentDataError("Invalid V1 augmentation report")
        version = "v1"
    elif isinstance(data.get("stats"), dict):
        rows = {
            key: value
            for key, value in data["stats"].items()
            if not key.startswith("__") and isinstance(value, dict)
        }
        version = "v2"
    else:
        raise ExperimentDataError("Augmentation report has an unsupported structure")

    class_rows = []
    bucket_totals: dict[str, int] = {}
    for class_name in sorted(rows):
        if not rows[class_name]:
            raise ExperimentDataError(f"Augmentation buckets are empty for {class_name}")
        buckets = {
            key: _as_count(value, f"augmentation.{class_name}.{key}")
            for key, value in rows[class_name].items()
        }
        total = sum(buckets.values())
        for bucket, count in buckets.items():
            bucket_totals[bucket] = bucket_totals.get(bucket, 0) + count
        class_rows.append(
            {"class_name": class_name, "buckets": buckets, "total": total}
        )

    return {
        "version": version,
        "train_total": sum(row["total"] for row in class_rows),
        "bucket_totals": bucket_totals,
        "class_augmentation": class_rows,
    }


def load_dataset_profile(
    outputs_dir: Path,
    dataset_id: str,
    version_id: str,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    registry = registry or _registry(outputs_dir)
    profile = _dataset_profile_entry(registry, dataset_id, version_id)
    base_path = _safe_output_path(outputs_dir, profile["base_dataprep"])
    augmentation_path = _safe_output_path(outputs_dir, profile["augmentation_dataprep"])
    base = _normalize_base_dataprep(_read_json(base_path))
    augmentation = _normalize_augmentation(_read_json(augmentation_path))
    return {
        **base,
        **augmentation,
        "sources": {
            "base_dataprep": profile["base_dataprep"],
            "augmentation_dataprep": profile["augmentation_dataprep"],
        },
    }


def _validate_history(history: Any) -> list[dict[str, float | int | None]]:
    if not isinstance(history, list) or not history:
        raise ExperimentDataError("Training history must be a non-empty list")
    normalized = []
    for index, row in enumerate(history):
        if not isinstance(row, dict):
            raise ExperimentDataError(f"History row {index} is not an object")
        normalized.append(
            {
                "epoch": int(_as_number(row.get("epoch"), f"history[{index}].epoch")),
                "train_loss": _as_number(
                    row.get("train_loss"), f"history[{index}].train_loss"
                ),
                "valid_unseen_loss": _as_number(
                    row.get("valid_unseen_loss"),
                    f"history[{index}].valid_unseen_loss",
                ),
                "valid_unseen_acc": _as_number(
                    row.get("valid_unseen_acc"), f"history[{index}].valid_unseen_acc"
                ),
                "elapsed_s": _as_optional_number(row.get("elapsed_s")),
            }
        )
    return normalized


def _training_summary(
    history: list[dict[str, Any]], selection_metric: str
) -> dict[str, Any]:
    min_loss_row = min(history, key=lambda row: row["valid_unseen_loss"])
    max_accuracy_row = max(history, key=lambda row: row["valid_unseen_acc"])
    selected_row = (
        max_accuracy_row
        if selection_metric == "maximum_valid_unseen_accuracy"
        else min_loss_row
    )
    elapsed_values = [row["elapsed_s"] for row in history if row["elapsed_s"] is not None]
    return {
        "completed_epochs": len(history),
        "selection_metric": selection_metric,
        "best_epoch": selected_row["epoch"],
        "best_valid_accuracy": selected_row["valid_unseen_acc"],
        "valid_loss_at_best_epoch": selected_row["valid_unseen_loss"],
        "minimum_valid_loss": min_loss_row["valid_unseen_loss"],
        "minimum_valid_loss_epoch": min_loss_row["epoch"],
        "maximum_valid_accuracy": max_accuracy_row["valid_unseen_acc"],
        "maximum_valid_accuracy_epoch": max_accuracy_row["epoch"],
        "total_elapsed_s": sum(elapsed_values) if elapsed_values else None,
    }


def _validated_evaluation(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ExperimentDataError("Evaluation file must contain an object")
    class_names = data.get("class_names")
    test = data.get("test")
    valid = data.get("valid_unseen")
    if not isinstance(class_names, list) or not class_names:
        raise ExperimentDataError("Evaluation is missing class_names")
    if not isinstance(test, dict) or not isinstance(valid, dict):
        raise ExperimentDataError("Evaluation is missing valid_unseen or test")

    report = test.get("classification_report")
    matrix = test.get("confusion_matrix")
    if not isinstance(report, dict) or not isinstance(matrix, list):
        raise ExperimentDataError("Test report or confusion matrix is missing")
    if len(matrix) != len(class_names) or any(
        not isinstance(row, list) or len(row) != len(class_names) for row in matrix
    ):
        raise ExperimentDataError("Confusion matrix dimensions do not match class_names")

    class_metrics = []
    supports = []
    f1_values = []
    for class_name in class_names:
        row = report.get(class_name)
        if not isinstance(row, dict):
            raise ExperimentDataError(f"Missing classification metrics for {class_name}")
        support = _as_count(row.get("support"), f"{class_name}.support")
        precision = _as_number(row.get("precision"), f"{class_name}.precision")
        recall = _as_number(row.get("recall"), f"{class_name}.recall")
        f1_score = _as_number(row.get("f1-score"), f"{class_name}.f1-score")
        supports.append(support)
        f1_values.append(f1_score)
        class_metrics.append(
            {
                "class_name": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1_score,
                "support": support,
            }
        )

    normalized_matrix = [
        [
            _as_count(value, f"confusion_matrix[{row_index}][{column_index}]")
            for column_index, value in enumerate(row)
        ]
        for row_index, row in enumerate(matrix)
    ]
    sample_count = sum(supports)
    matrix_count = sum(sum(row) for row in normalized_matrix)
    declared_samples = test.get("samples")
    if declared_samples is not None and _as_count(declared_samples, "test.samples") != sample_count:
        raise ExperimentDataError(
            f"Declared test samples {declared_samples} do not match report support {sample_count}"
        )
    if matrix_count != sample_count:
        raise ExperimentDataError(
            f"Confusion matrix total {matrix_count} does not match report support {sample_count}"
        )

    correct = sum(normalized_matrix[index][index] for index in range(len(class_names)))
    computed_accuracy = correct / sample_count if sample_count else 0.0
    declared_accuracy = _as_number(test.get("accuracy"), "test.accuracy")
    if abs(computed_accuracy - declared_accuracy) > 0.0005:
        raise ExperimentDataError(
            f"Test accuracy {declared_accuracy} does not match confusion matrix {computed_accuracy}"
        )

    macro_f1 = sum(f1_values) / len(f1_values)
    weighted_f1 = (
        sum(metric["f1"] * metric["support"] for metric in class_metrics) / sample_count
        if sample_count
        else 0.0
    )

    top_errors = []
    for true_index, row in enumerate(normalized_matrix):
        row_total = sum(row)
        for predicted_index, value in enumerate(row):
            count = value
            if true_index == predicted_index or count <= 0:
                continue
            top_errors.append(
                {
                    "true_class": class_names[true_index],
                    "predicted_class": class_names[predicted_index],
                    "count": count,
                    "rate": count / row_total if row_total else 0.0,
                }
            )
    top_errors.sort(key=lambda item: item["count"], reverse=True)

    return {
        "class_names": class_names,
        "test": {
            "accuracy": computed_accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "samples": sample_count,
            "loss": _as_optional_number(test.get("loss")),
        },
        "valid_unseen": {
            "accuracy": _as_number(valid.get("accuracy"), "valid_unseen.accuracy"),
            "loss": _as_optional_number(valid.get("loss")),
            "samples": _as_count(valid.get("samples"), "valid_unseen.samples")
            if valid.get("samples") is not None
            else None,
        },
        "class_metrics": class_metrics,
        "confusion_matrix": normalized_matrix,
        "top_errors": top_errors[:12],
    }


def load_experiment(outputs_dir: Path, experiment_id: str) -> dict[str, Any]:
    registry = _registry(outputs_dir)
    experiment = _lookup(registry["experiments"], experiment_id, "experiment")
    model = _lookup(registry.get("models", []), experiment["model"], "model")
    dataset = _lookup(registry.get("datasets", []), experiment["dataset"], "dataset")
    version = _lookup(registry.get("versions", []), experiment["version"], "version")

    evaluation_path = _safe_output_path(outputs_dir, experiment["evaluation"])
    history_path = _safe_output_path(outputs_dir, experiment["history"])
    evaluation = _validated_evaluation(_read_json(evaluation_path))
    history = _validate_history(_read_json(history_path))
    dataset_profile = load_dataset_profile(
        outputs_dir, experiment["dataset"], experiment["version"], registry
    )
    training = _training_summary(history, experiment["selection_metric"])

    # Load optional hyperparameters
    hyperparameters = None
    if "metrics" in experiment:
        metrics_file_path = _safe_output_path(outputs_dir, experiment["metrics"])
        if metrics_file_path.is_file():
            try:
                metrics_data = json.loads(metrics_file_path.read_text(encoding="utf-8"))
                hyperparameters = metrics_data.get("hyperparameters")
            except Exception:
                pass

    if not hyperparameters:
        try:
            eval_path_obj = Path(experiment["evaluation"])
            manifest_path = outputs_dir / eval_path_obj.parent / "experiment_manifest.json"
            if manifest_path.is_file():
                manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                hyperparameters = manifest_data.get("hyperparameters")
        except Exception:
            pass
                
    if not hyperparameters and experiment.get("model") == "yolo":
        eval_path_obj = Path(experiment["evaluation"])
        args_path = outputs_dir / eval_path_obj.parent / "args.yaml"
        if args_path.is_file():
            try:
                yolo_config = {}
                with open(args_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or ":" not in line:
                            continue
                        k, v = line.split(":", 1)
                        k, v = k.strip(), v.strip()
                        if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
                            v = v[1:-1]
                        yolo_config[k] = v
                hyperparameters = yolo_config
            except Exception:
                pass

    weakest = min(evaluation["class_metrics"], key=lambda row: row["f1"])
    largest_error = evaluation["top_errors"][0] if evaluation["top_errors"] else None
    generalization_gap = (
        evaluation["valid_unseen"]["accuracy"] - evaluation["test"]["accuracy"]
    )

    insights = [
        {
            "kind": "weakest_class",
            "message": f"Lớp có F1 thấp nhất là {weakest['class_name']} ({weakest['f1'] * 100:.1f}%).",
        },
        {
            "kind": "generalization_gap",
            "message": (
                "Validation cao hơn test "
                f"{abs(generalization_gap) * 100:.2f} điểm %."
                if generalization_gap >= 0
                else "Test cao hơn validation "
                f"{abs(generalization_gap) * 100:.2f} điểm %."
            ),
        },
    ]
    if largest_error:
        insights.append(
            {
                "kind": "top_error",
                "message": (
                    f"Nhầm nhiều nhất: {largest_error['true_class']} -> "
                    f"{largest_error['predicted_class']} ({largest_error['count']} ảnh)."
                ),
            }
        )

    return {
        "id": experiment["id"],
        "model": {"id": model["id"], "label": model["label"]},
        "dataset": {
            "id": dataset["id"],
            "label": dataset["label"],
            "description": dataset.get("description"),
        },
        "version": {
            "id": version["id"],
            "label": version["label"],
            "description": version.get("description"),
        },
        "metadata_source": experiment.get("metadata_source"),
        "evaluation": evaluation,
        "training": training,
        "history": history,
        "dataset_profile": dataset_profile,
        "insights": insights,
        "hyperparameters": hyperparameters,
        "sources": {
            "evaluation": experiment["evaluation"],
            "history": experiment["history"],
            **dataset_profile["sources"],
        },
    }


def compare_experiments(
    outputs_dir: Path, first_id: str, second_id: str
) -> dict[str, Any]:
    first = load_experiment(outputs_dir, first_id)
    second = load_experiment(outputs_dir, second_id)
    changed_dimensions = [
        dimension
        for dimension in ["model", "dataset", "version"]
        if first[dimension]["id"] != second[dimension]["id"]
    ]

    warnings = []
    if len(changed_dimensions) > 1:
        warnings.append(
            "Đang thay đổi nhiều hơn một yếu tố; không thể quy chênh lệch cho một nguyên nhân duy nhất."
        )
    first_samples = first["evaluation"]["test"]["samples"]
    second_samples = second["evaluation"]["test"]["samples"]
    if first_samples != second_samples:
        warnings.append(
            f"Hai tập test có số mẫu khác nhau ({first_samples} và {second_samples})."
        )

    first_metrics = first["evaluation"]["test"]
    second_metrics = second["evaluation"]["test"]
    deltas = {
        key: second_metrics[key] - first_metrics[key]
        for key in ["accuracy", "macro_f1", "weighted_f1"]
    }

    first_classes = {
        item["class_name"]: item for item in first["evaluation"]["class_metrics"]
    }
    second_classes = {
        item["class_name"]: item for item in second["evaluation"]["class_metrics"]
    }
    shared_classes = sorted(set(first_classes) & set(second_classes))
    class_deltas = [
        {
            "class_name": class_name,
            "first_f1": first_classes[class_name]["f1"],
            "second_f1": second_classes[class_name]["f1"],
            "delta_f1": second_classes[class_name]["f1"] - first_classes[class_name]["f1"],
            "first_support": first_classes[class_name]["support"],
            "second_support": second_classes[class_name]["support"],
        }
        for class_name in shared_classes
    ]

    return {
        "first": first,
        "second": second,
        "changed_dimensions": changed_dimensions,
        "warnings": warnings,
        "deltas": deltas,
        "class_deltas": class_deltas,
    }
