"""Input schema and leakage-safe dataset splitting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Revenue"
NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]
CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class SchemaError(ValueError):
    """Raised when the incoming table cannot satisfy the model contract."""


def _binary_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    values = series.astype(str).str.strip().str.lower()
    mapped = values.map({"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0})
    if mapped.isna().any():
        bad = sorted(values[mapped.isna()].unique().tolist())
        raise SchemaError(f"{TARGET} must be binary; unsupported values: {bad[:5]}")
    return mapped.astype(int)


def validate_frame(frame: pd.DataFrame, *, require_target: bool = True) -> pd.DataFrame:
    required = FEATURES + ([TARGET] if require_target else [])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    if frame.empty:
        raise SchemaError("dataset is empty")

    clean = frame.copy()
    for column in NUMERIC_FEATURES:
        converted = pd.to_numeric(clean[column], errors="coerce")
        if converted.isna().sum() > clean[column].isna().sum():
            raise SchemaError(f"{column} contains non-numeric values")
        clean[column] = converted
    for column in ["Administrative", "Informational", "ProductRelated"]:
        if (clean[column].dropna() < 0).any():
            raise SchemaError(f"{column} cannot be negative")
    for column in ["BounceRates", "ExitRates", "SpecialDay"]:
        values = clean[column].dropna()
        if ((values < 0) | (values > 1)).any():
            raise SchemaError(f"{column} must be between 0 and 1")
    if require_target:
        clean[TARGET] = _binary_target(clean[TARGET])
        if clean[TARGET].nunique() != 2:
            raise SchemaError(f"{TARGET} must contain both classes")
    return clean


def load_data(path: str | Path, *, require_target: bool = True) -> pd.DataFrame:
    return validate_frame(pd.read_csv(path), require_target=require_target)


def split_data(
    frame: pd.DataFrame, *, seed: int = 20250607
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if TARGET not in frame:
        raise SchemaError(f"{TARGET} is required for splitting")
    class_counts = frame[TARGET].value_counts()
    if set(class_counts.index) != {0, 1} or int(class_counts.min()) < 5:
        raise SchemaError(
            "stratified 60/20/20 split requires both classes and at least 5 rows per class"
        )
    train_validation, test = train_test_split(
        frame,
        test_size=0.20,
        random_state=seed,
        stratify=frame[TARGET],
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.25,
        random_state=seed,
        stratify=train_validation[TARGET],
    )
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )
