"""Run a reproducible synthetic sensitivity experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    split_data,
    validate_frame,
)
from .generate_smoke_data import generate_smoke_frame
from .modeling import candidate_models, classification_metrics


def inject_missing_values(frame, *, rate: float, seed: int):
    """Inject missing feature values without changing the target."""
    if not 0 <= rate < 0.5:
        raise ValueError("missing_rate must be in [0, 0.5)")
    if rate == 0:
        return frame.copy()
    result = frame.copy()
    rng = np.random.default_rng(seed)
    for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        result.loc[rng.random(len(result)) < rate, column] = np.nan
    return result


def inject_unseen_categories(frame, *, rate: float, seed: int):
    """Replace a share of categories after splitting to test unknown-value handling."""
    if not 0 <= rate < 0.5:
        raise ValueError("unseen_category_rate must be in [0, 0.5)")
    result = frame.copy()
    if rate == 0:
        return result
    rng = np.random.default_rng(seed)
    for column in CATEGORICAL_FEATURES:
        mask = rng.random(len(result)) < rate
        result.loc[mask, column] = f"__UNSEEN_{column}__"
    return result


def run_experiment(
    *,
    rows: int,
    data_seed: int,
    split_seed: int,
    budget_fraction: float,
    hypothesis: str,
    missing_rate: float = 0.0,
    unseen_category_rate: float = 0.0,
    baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    """Train every candidate and return deterministic validation and test metrics."""
    if not hypothesis.strip():
        raise ValueError("hypothesis must not be empty")
    if not 0 < budget_fraction <= 1:
        raise ValueError("budget_fraction must be in (0, 1]")

    generated = generate_smoke_frame(rows=rows, seed=data_seed)
    frame = validate_frame(
        inject_missing_values(generated, rate=missing_rate, seed=data_seed + 10_003)
    )
    train, validation, test = split_data(frame, seed=split_seed)
    validation = inject_unseen_categories(
        validation,
        rate=unseen_category_rate,
        seed=data_seed + split_seed + 20_003,
    )
    test = inject_unseen_categories(
        test,
        rate=unseen_category_rate,
        seed=data_seed + split_seed + 30_007,
    )
    validation_metrics: dict[str, dict[str, object]] = {}
    fitted_models = {}

    for name, model in candidate_models(split_seed).items():
        model.fit(train[FEATURES], train[TARGET])
        scores = model.predict_proba(validation[FEATURES])[:, 1]
        validation_metrics[name] = classification_metrics(
            validation[TARGET], scores, budget_fraction
        )
        fitted_models[name] = model

    selected_model = max(
        validation_metrics,
        key=lambda name: float(validation_metrics[name]["pr_auc"]),
    )
    test_scores = fitted_models[selected_model].predict_proba(test[FEATURES])[:, 1]
    test_metrics = classification_metrics(test[TARGET], test_scores, budget_fraction)

    result: dict[str, object] = {
        "schema_version": 1,
        "experiment": "synthetic_sensitivity",
        "hypothesis": hypothesis.strip(),
        "parameters": {
            "rows": rows,
            "data_seed": data_seed,
            "split_seed": split_seed,
            "budget_fraction": budget_fraction,
            "missing_rate": missing_rate,
            "unseen_category_rate": unseen_category_rate,
        },
        "dataset": {
            "mode": "synthetic",
            "positive_rate": float(frame[TARGET].mean()),
            "missing_feature_values": int(frame[FEATURES].isna().sum().sum()),
            "unseen_category_values": int(
                validation[CATEGORICAL_FEATURES]
                .astype(str)
                .apply(lambda column: column.str.startswith("__UNSEEN_").sum())
                .sum()
                + test[CATEGORICAL_FEATURES]
                .astype(str)
                .apply(lambda column: column.str.startswith("__UNSEEN_").sum())
                .sum()
            ),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
        },
        "selection": {
            "primary_metric": "pr_auc",
            "selected_model": selected_model,
        },
        "validation": {"models": validation_metrics},
        "test": test_metrics,
    }
    if baseline is not None:
        baseline_test = baseline.get("test")
        baseline_selection = baseline.get("selection")
        if not isinstance(baseline_test, dict) or not isinstance(baseline_selection, dict):
            raise ValueError("baseline does not satisfy the experiment schema")
        compared_metrics = ("pr_auc", "roc_auc", "precision", "recall", "f1")
        result["comparison"] = {
            "baseline_model": baseline_selection["selected_model"],
            "model_changed": selected_model != baseline_selection["selected_model"],
            "test_delta": {
                metric: float(test_metrics[metric]) - float(baseline_test[metric])
                for metric in compared_metrics
            },
        }
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--rows", type=int, default=320)
    parser.add_argument("--data-seed", type=int, default=20250607)
    parser.add_argument("--split-seed", type=int, default=20250607)
    parser.add_argument("--budget-fraction", type=float, default=0.20)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--missing-rate", type=float, default=0.0)
    parser.add_argument("--unseen-category-rate", type=float, default=0.0)
    args = parser.parse_args(argv)

    baseline = None
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    result = run_experiment(
        rows=args.rows,
        data_seed=args.data_seed,
        split_seed=args.split_seed,
        budget_fraction=args.budget_fraction,
        hypothesis=args.hypothesis,
        missing_rate=args.missing_rate,
        unseen_category_rate=args.unseen_category_rate,
        baseline=baseline,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"experiment result written to {args.output}")


if __name__ == "__main__":
    main()
