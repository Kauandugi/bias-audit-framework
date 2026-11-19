"""Generate reproducible reports from frozen MLLM judge outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from biasauditfw.statistics import (
    judge_clip_convergence,
    judge_interrater_report,
    judge_repeatability_report,
    pairwise_judge_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointwise", type=Path, nargs="+", required=True)
    parser.add_argument("--pairwise", type=Path, nargs="+", required=True)
    parser.add_argument("--images-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    pointwise = pd.concat(
        [pd.read_csv(path) for path in args.pointwise], ignore_index=True
    )
    pairwise = pd.concat(
        [pd.read_csv(path) for path in args.pairwise], ignore_index=True
    )
    images = pd.concat(
        [pd.read_csv(path) for path in args.images_csv], ignore_index=True
    )
    if images["imagem_id"].duplicated().any():
        raise ValueError("Image inventories contain duplicate imagem_id values")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    judge_repeatability_report(pointwise).to_csv(
        args.output_dir / "judge_repeatability.csv", index=False
    )
    judge_interrater_report(pointwise).to_csv(
        args.output_dir / "judge_interrater_agreement.csv", index=False
    )
    judge_clip_convergence(pointwise, images).to_csv(
        args.output_dir / "judge_clip_convergence.csv", index=False
    )
    pairwise_judge_report(pairwise, images).to_csv(
        args.output_dir / "judge_pairwise_report.csv", index=False
    )

    eligible = pointwise.loc[pointwise["eligible"].astype(bool)].copy()
    eligible["judge_id"] = (
        eligible["provider"].astype(str) + ":" + eligible["model_id"].astype(str)
    )
    descriptive = (
        eligible.groupby(["judge_id", "dataset_id"], as_index=False)
        .agg(
            n_images=("imagem_id", "nunique"),
            racial_median=("racial_score", "median"),
            racial_q1=("racial_score", lambda values: values.quantile(0.25)),
            racial_q3=("racial_score", lambda values: values.quantile(0.75)),
            gender_median=("gender_score", "median"),
            gender_q1=("gender_score", lambda values: values.quantile(0.25)),
            gender_q3=("gender_score", lambda values: values.quantile(0.75)),
        )
    )
    descriptive.to_csv(
        args.output_dir / "judge_pointwise_descriptive.csv", index=False
    )
    print(f"MLLM reports written to {args.output_dir}")


if __name__ == "__main__":
    main()
