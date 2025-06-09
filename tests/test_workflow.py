from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from online_shopper.data import SchemaError, split_data, validate_frame
from online_shopper.evaluate import main as evaluate
from online_shopper.experiment import run_experiment
from online_shopper.generate_smoke_data import generate_smoke_frame
from online_shopper.modeling import classification_metrics, select_for_budget
from online_shopper.predict import main as predict
from online_shopper.train import main as train


class ShopperWorkflowTest(unittest.TestCase):
    def test_synthetic_experiment_records_selection_contract(self) -> None:
        baseline = run_experiment(
            rows=120,
            data_seed=17,
            split_seed=23,
            budget_fraction=0.15,
            hypothesis="Проверить полный контракт эксперимента",
        )
        result = run_experiment(
            rows=120,
            data_seed=17,
            split_seed=29,
            budget_fraction=0.15,
            hypothesis="Сравнить другое разбиение с базовым",
            baseline=baseline,
        )

        self.assertEqual(result["dataset"]["mode"], "synthetic")
        self.assertIn(result["selection"]["selected_model"], result["validation"]["models"])
        self.assertEqual(result["test"]["selected_fraction"], 1 / 6)
        self.assertEqual(
            set(result["comparison"]["test_delta"]),
            {"pr_auc", "roc_auc", "precision", "recall", "f1"},
        )

    def test_contact_budget_is_exact_when_scores_are_tied(self) -> None:
        scores = np.full(10, 0.5)
        selected = select_for_budget(scores, 0.20)
        metrics = classification_metrics(np.array([0, 1] * 5), scores, 0.20)

        self.assertEqual(selected.sum(), 2)
        self.assertEqual(np.flatnonzero(selected).tolist(), [0, 1])
        self.assertEqual(metrics["selected_fraction"], 0.20)

    def test_schema_and_stratified_split(self) -> None:
        frame = validate_frame(generate_smoke_frame(120))
        pd.testing.assert_frame_equal(generate_smoke_frame(120), generate_smoke_frame(120))
        train_frame, validation, test = split_data(frame)
        self.assertEqual(len(frame), len(train_frame) + len(validation) + len(test))
        with self.assertRaises(SchemaError):
            validate_frame(frame.drop(columns=["Revenue"]))

        too_small = frame.groupby("Revenue", group_keys=False).head(4)
        with self.assertRaisesRegex(SchemaError, "at least 5 rows per class"):
            split_data(too_small)

    def test_end_to_end_without_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "smoke.csv"
            artifact = root / "model.joblib"
            validation = root / "validation.json"
            metrics = root / "metrics.json"
            errors = root / "errors.csv"
            predictions = root / "predictions.csv"
            frame = generate_smoke_frame(160)
            frame.to_csv(data_path, index=False)
            train(
                ["--data", str(data_path), "--artifact", str(artifact), "--report", str(validation)]
            )
            evaluate(
                [
                    "--data",
                    str(data_path),
                    "--artifact",
                    str(artifact),
                    "--metrics",
                    str(metrics),
                    "--errors",
                    str(errors),
                ]
            )
            inference = root / "inference.csv"
            frame.drop(columns=["Revenue"]).head(7).to_csv(inference, index=False)
            predict(
                [
                    "--data",
                    str(inference),
                    "--artifact",
                    str(artifact),
                    "--output",
                    str(predictions),
                ]
            )
            self.assertIn("pr_auc", json.loads(metrics.read_text(encoding="utf-8")))
            self.assertEqual(len(pd.read_csv(predictions)), 7)


if __name__ == "__main__":
    unittest.main()
