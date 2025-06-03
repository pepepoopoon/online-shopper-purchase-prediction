"""Evaluate the frozen shopper model and write row-level errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from .data import FEATURES, TARGET, load_data, split_data
from .modeling import classification_metrics, select_for_budget


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--metrics", type=Path, default=Path("reports/test_metrics.json"))
    parser.add_argument("--errors", type=Path, default=Path("reports/test_errors.csv"))
    args = parser.parse_args(argv)

    artifact = joblib.load(args.artifact)
    _, _, test = split_data(load_data(args.data), seed=int(artifact["seed"]))
    scores = artifact["model"].predict_proba(test[FEATURES])[:, 1]
    budget_fraction = float(artifact["budget_fraction"])
    labels = select_for_budget(scores, budget_fraction).astype(int)
    metrics = classification_metrics(test[TARGET], scores, budget_fraction)
    metrics["model_name"] = artifact["model_name"]
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    errors = test[["VisitorType", "Month", TARGET]].copy()
    errors["score"] = scores
    errors["prediction"] = labels
    errors["error_type"] = np.select(
        [(test[TARGET] == 0) & (labels == 1), (test[TARGET] == 1) & (labels == 0)],
        ["false_positive", "false_negative"],
        default="correct",
    )
    errors = errors[errors["error_type"] != "correct"].sort_values("score", ascending=False)
    args.errors.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(args.errors, index=False)
    print(f"test metrics written to {args.metrics}; errors written to {args.errors}")


if __name__ == "__main__":
    main()
