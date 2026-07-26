"""Export descriptive and inferential reports from schema 2.0 outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .statistics import (
    PRIMARY_METRICS,
    face_diversity_by_image,
    inference_eligibility,
    mann_whitney_reports,
    per_model_mann_whitney,
    wilcoxon_reports,
)


def descriptive_report(images: pd.DataFrame) -> pd.DataFrame:
    """Summarize both semantic margins without requiring a manifest."""

    group_columns = [
        column for column in ("modelo_ia", "tipo_prompt") if column in images
    ]
    rows: list[dict[str, Any]] = []
    groupers: list[tuple[str, pd.DataFrame]] = [("overall", images)]
    for column in group_columns:
        groupers.extend(
            (f"{column}={value}", subset)
            for value, subset in images.groupby(column, dropna=False)
        )
    for group, subset in groupers:
        for metric in PRIMARY_METRICS:
            values = pd.to_numeric(subset[metric], errors="coerce").dropna()
            rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "n": len(values),
                    "mean": values.mean(),
                    "standard_deviation": values.std(ddof=1),
                    "median": values.median(),
                    "q1": values.quantile(0.25),
                    "q3": values.quantile(0.75),
                    "minimum": values.min(),
                    "maximum": values.max(),
                }
            )
    return pd.DataFrame(rows)


def export_statistical_reports(
    images: pd.DataFrame,
    faces: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """Always export descriptive evidence and conditionally run inference."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptive_report(images).to_csv(
        output_dir / "descriptive_diversity.csv", index=False
    )
    face_diversity_by_image(faces).to_csv(
        output_dir / "deepface_simpson_by_image.csv", index=False
    )

    inferential_files = (
        "mann_whitney_primary.csv",
        "wilcoxon_sensitivity.csv",
        "mann_whitney_by_model.csv",
    )
    for filename in inferential_files:
        (output_dir / filename).unlink(missing_ok=True)

    eligible, reason = inference_eligibility(images)
    status = {"eligible": eligible, "reason": reason}
    (output_dir / "inference_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if eligible:
        mann_whitney_reports(images).to_csv(
            output_dir / "mann_whitney_primary.csv", index=False
        )
        wilcoxon_reports(images).to_csv(
            output_dir / "wilcoxon_sensitivity.csv", index=False
        )
        per_model_mann_whitney(images).to_csv(
            output_dir / "mann_whitney_by_model.csv", index=False
        )
    return status
