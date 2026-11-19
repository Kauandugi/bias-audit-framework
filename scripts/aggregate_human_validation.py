"""Aggregate private human responses without exporting participant-level data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from biasauditfw.statistics import (
    aggregate_human_ratings,
    human_judge_convergence,
    ordinal_krippendorff_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-private-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-pointwise", type=Path)
    args = parser.parse_args()

    raw = pd.read_csv(args.raw_private_csv)
    aggregate = aggregate_human_ratings(raw)
    agreement = ordinal_krippendorff_report(raw)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(args.output_dir / "human_ratings_aggregate.csv", index=False)
    agreement.to_csv(args.output_dir / "human_interrater_agreement.csv", index=False)
    if args.judge_pointwise:
        pointwise = pd.read_csv(args.judge_pointwise)
        human_judge_convergence(aggregate, pointwise).to_csv(
            args.output_dir / "human_judge_convergence.csv", index=False
        )
    print("Only aggregate human artifacts were written.")


if __name__ == "__main__":
    main()
