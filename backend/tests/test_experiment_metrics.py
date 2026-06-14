from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.experiment_metrics import (  # noqa: E402
    build_catalog,
    compare_experiments,
    load_dataset_profile,
    load_experiment,
)


class ExperimentMetricsTests(unittest.TestCase):
    outputs_dir = PROJECT_ROOT / "outputs"

    def test_registered_experiments_are_internally_consistent(self) -> None:
        catalog = build_catalog(self.outputs_dir)
        ready_ids = [
            item["id"] for item in catalog["combinations"] if item["available"]
        ]

        self.assertEqual(len(ready_ids), 3)
        for experiment_id in ready_ids:
            experiment = load_experiment(self.outputs_dir, experiment_id)
            evaluation = experiment["evaluation"]
            matrix_total = sum(sum(row) for row in evaluation["confusion_matrix"])
            support_total = sum(row["support"] for row in evaluation["class_metrics"])

            self.assertEqual(matrix_total, evaluation["test"]["samples"])
            self.assertEqual(support_total, evaluation["test"]["samples"])

    def test_all_dataset_profiles_use_recorded_counts(self) -> None:
        expected_train_totals = {
            ("raw_original", "v1"): 75620,
            ("raw_original", "v2"): 75620,
            ("raw_cleaning", "v1"): 73900,
            ("raw_cleaning", "v2"): 73900,
        }

        for key, expected_total in expected_train_totals.items():
            profile = load_dataset_profile(self.outputs_dir, *key)
            self.assertEqual(profile["train_total"], expected_total)

    def test_comparison_delta_is_second_minus_first(self) -> None:
        comparison = compare_experiments(
            self.outputs_dir,
            "resnet50__raw_original__v1",
            "vit__raw_original__v1",
        )
        expected = (
            comparison["second"]["evaluation"]["test"]["accuracy"]
            - comparison["first"]["evaluation"]["test"]["accuracy"]
        )

        self.assertAlmostEqual(comparison["deltas"]["accuracy"], expected)


if __name__ == "__main__":
    unittest.main()
