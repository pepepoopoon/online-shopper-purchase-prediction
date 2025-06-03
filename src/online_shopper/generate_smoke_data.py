"""Create deterministic synthetic data for execution checks only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_smoke_frame(rows: int = 320, seed: int = 20250607) -> pd.DataFrame:
    if rows < 80:
        raise ValueError("rows must be at least 80")
    rng = np.random.default_rng(seed)
    product_pages = rng.poisson(28, rows)
    page_values = np.maximum(0, rng.gamma(1.4, 18, rows) - 12)
    returning = rng.choice(["Returning_Visitor", "New_Visitor", "Other"], rows, p=[0.7, 0.27, 0.03])
    weekend = rng.choice([True, False], rows, p=[0.28, 0.72])
    logits = (
        -3.2
        + 0.055 * page_values
        + 0.018 * product_pages
        + 0.55 * (returning == "Returning_Visitor")
    )
    probability = 1 / (1 + np.exp(-logits))
    revenue = rng.binomial(1, probability).astype(bool)
    return pd.DataFrame(
        {
            "Administrative": rng.poisson(2, rows),
            "Administrative_Duration": rng.gamma(1.4, 55, rows),
            "Informational": rng.poisson(0.7, rows),
            "Informational_Duration": rng.gamma(1.2, 30, rows),
            "ProductRelated": product_pages,
            "ProductRelated_Duration": product_pages * rng.gamma(1.5, 22, rows),
            "BounceRates": rng.beta(1.2, 14, rows),
            "ExitRates": rng.beta(1.5, 9, rows),
            "PageValues": page_values,
            "SpecialDay": rng.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], rows),
            "Month": rng.choice(["Feb", "Mar", "May", "June", "Jul", "Aug", "Nov", "Dec"], rows),
            "OperatingSystems": rng.integers(1, 5, rows).astype(str),
            "Browser": rng.integers(1, 9, rows).astype(str),
            "Region": rng.integers(1, 10, rows).astype(str),
            "TrafficType": rng.integers(1, 14, rows).astype(str),
            "VisitorType": returning,
            "Weekend": weekend.astype(str),
            "Revenue": revenue,
        }
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/smoke.csv"))
    parser.add_argument("--rows", type=int, default=320)
    parser.add_argument("--seed", type=int, default=20250607)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_smoke_frame(args.rows, args.seed).to_csv(args.output, index=False)
    print(f"synthetic smoke data written to {args.output}")


if __name__ == "__main__":
    main()
