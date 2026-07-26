"""Validate a complete schema 2.0 Colab execution independently."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biasauditfw.contracts import validate_contract
from biasauditfw.statistics import mann_whitney_reports, wilcoxon_reports


REQUIRED_FILES = {
    "audit_images.csv",
    "audit_faces.csv",
    "ingestion_report.csv",
    "run_metadata.json",
    "descriptive_diversity.csv",
    "deepface_simpson_by_image.csv",
    "inference_status.json",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _compare_reports(
    calculated: pd.DataFrame,
    exported: pd.DataFrame,
    keys: list[str],
    numeric_columns: list[str],
) -> None:
    merged = calculated.merge(
        exported,
        on=keys,
        suffixes=("_calculated", "_exported"),
        validate="one_to_one",
    )
    require(len(merged) == len(calculated), "Exported statistical rows are incomplete")
    for column in numeric_columns:
        require(
            np.allclose(
                merged[f"{column}_calculated"],
                merged[f"{column}_exported"],
                equal_nan=True,
            ),
            f"Exported {column} differs from independent recalculation",
        )


def validate_outputs(
    output_dir: Path,
    expected_images: int | None = None,
    require_cuda: bool = False,
) -> dict[str, Any]:
    existing = {path.name for path in output_dir.iterdir() if path.is_file()}
    missing = REQUIRED_FILES - existing
    require(not missing, f"Missing files: {sorted(missing)}")

    images = pd.read_csv(output_dir / "audit_images.csv")
    faces = pd.read_csv(output_dir / "audit_faces.csv")
    metadata = json.loads(
        (output_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    inference = json.loads(
        (output_dir / "inference_status.json").read_text(encoding="utf-8")
    )

    validate_contract(images, faces, expected_images)
    require(metadata.get("schema_version") == "2.0", "Unexpected schema version")
    require(
        metadata.get("clip_scoring")
        == "L2-normalized cosine prototype margins",
        "CLIP scoring metadata is inconsistent",
    )
    if require_cuda:
        require(metadata.get("cuda_available") is True, "CUDA was not recorded")
        require("T4" in str(metadata.get("gpu_name")), "Tesla T4 was not recorded")

    if inference.get("eligible"):
        required_inference = {
            "mann_whitney_primary.csv",
            "wilcoxon_sensitivity.csv",
            "mann_whitney_by_model.csv",
        }
        require(
            required_inference.issubset(existing),
            "Eligible execution is missing inferential reports",
        )
        _compare_reports(
            mann_whitney_reports(images),
            pd.read_csv(output_dir / "mann_whitney_primary.csv"),
            ["metric"],
            ["u_statistic", "p_value", "p_value_holm", "rank_biserial"],
        )
        _compare_reports(
            wilcoxon_reports(images),
            pd.read_csv(output_dir / "wilcoxon_sensitivity.csv"),
            ["metric"],
            ["w_statistic", "p_value", "p_value_holm"],
        )
    else:
        require(
            not {
                "mann_whitney_primary.csv",
                "wilcoxon_sensitivity.csv",
            }
            & existing,
            "Ineligible dataset must not contain inferential reports",
        )

    return {
        "schema_version": metadata["schema_version"],
        "images": len(images),
        "faces": len(faces),
        "inference_eligible": bool(inference.get("eligible")),
        "inference_reason": inference.get("reason"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-images", type=int)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    summary = validate_outputs(
        args.output_dir,
        expected_images=args.expected_images,
        require_cuda=args.require_cuda,
    )
    print("Validation passed:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
