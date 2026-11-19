"""Independently validate frozen detector artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from biasauditfw.validation import (
    average_precision_at_iou,
    bootstrap_detector,
    bootstrap_detector_disagreement,
    compare_detectors,
    evaluate_detector,
)


def _load_metadata(output_dir: Path) -> dict[str, object]:
    path = output_dir / "detector_validation_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing detector metadata: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Detector metadata must contain a JSON object")
    return metadata


def _validate_synthetic(
    output_dir: Path,
    expected_images: int,
    metadata: dict[str, object],
) -> dict[str, object]:
    required = {
        "synthetic_retinaface_boxes.csv",
        "synthetic_mtcnn_boxes.csv",
        "detector_disagreement_per_image.csv",
        "detector_disagreement_summary.csv",
        "detector_disagreement_bootstrap_ci.csv",
    }
    missing = sorted(name for name in required if not (output_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"Missing synthetic detector artifacts: {missing}")

    retinaface = pd.read_csv(output_dir / "synthetic_retinaface_boxes.csv")
    mtcnn = pd.read_csv(output_dir / "synthetic_mtcnn_boxes.csv")
    saved_per_image = pd.read_csv(
        output_dir / "detector_disagreement_per_image.csv"
    )
    saved_summary = pd.read_csv(output_dir / "detector_disagreement_summary.csv")
    saved_ci = pd.read_csv(
        output_dir / "detector_disagreement_bootstrap_ci.csv"
    ).sort_values("metric", ignore_index=True)

    if len(saved_per_image) != expected_images:
        raise AssertionError(
            f"Expected {expected_images} images, found {len(saved_per_image)}"
        )
    if saved_per_image["imagem_id"].duplicated().any():
        raise AssertionError("Duplicate imagem_id values in synthetic inventory")
    images = saved_per_image[["imagem_id"]].copy()
    known_ids = set(images["imagem_id"].astype(str))
    for label, table in (("retinaface", retinaface), ("mtcnn", mtcnn)):
        unknown = set(table["imagem_id"].astype(str)) - known_ids
        if unknown:
            raise AssertionError(f"{label} contains unknown imagem_id values")

    per_image, summary = compare_detectors(retinaface, mtcnn, images)
    ci = bootstrap_detector_disagreement(per_image).sort_values(
        "metric", ignore_index=True
    )
    saved_per_image = saved_per_image.sort_values("imagem_id", ignore_index=True)
    per_image = per_image.sort_values("imagem_id", ignore_index=True)
    pd.testing.assert_frame_equal(
        saved_per_image,
        per_image,
        check_dtype=False,
        atol=1e-12,
        rtol=1e-12,
    )
    pd.testing.assert_frame_equal(
        saved_summary,
        summary,
        check_dtype=False,
        atol=1e-12,
        rtol=1e-12,
    )
    pd.testing.assert_frame_equal(
        saved_ci,
        ci,
        check_dtype=False,
        atol=1e-12,
        rtol=1e-12,
    )
    if metadata.get("image_count") != expected_images:
        raise AssertionError("Metadata image_count does not match the inventory")
    row = summary.iloc[0]
    return {
        "images": len(images),
        "retinaface_boxes": len(retinaface),
        "mtcnn_boxes": len(mtcnn),
        "exact_count_agreement_rate": row["exact_count_agreement_rate"],
        "mean_absolute_count_difference": row["mean_absolute_count_difference"],
        "mean_matched_iou": row["mean_matched_iou"],
        "mean_unmatched_rate": row["mean_unmatched_rate"],
    }


def _validate_wider(
    output_dir: Path,
    expected_images: int,
    metadata: dict[str, object],
) -> dict[str, object]:


    required = {
        "wider_images.csv",
        "wider_ground_truth.csv",
        "wider_retinaface_predictions.csv",
        "wider_detector_per_image.csv",
        "wider_detector_summary.csv",
        "wider_detector_bootstrap_ci.csv",
        "detector_validation_metadata.json",
    }
    missing = sorted(name for name in required if not (output_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"Missing WIDER artifacts: {missing}")

    images = pd.read_csv(output_dir / "wider_images.csv")
    truth = pd.read_csv(output_dir / "wider_ground_truth.csv")
    predictions = pd.read_csv(
        output_dir / "wider_retinaface_predictions.csv"
    )
    saved_per_image = pd.read_csv(
        output_dir / "wider_detector_per_image.csv"
    ).sort_values("imagem_id", ignore_index=True)
    saved_summary = pd.read_csv(
        output_dir / "wider_detector_summary.csv"
    ).sort_values("grupo", ignore_index=True)
    saved_ci = pd.read_csv(
        output_dir / "wider_detector_bootstrap_ci.csv"
    ).sort_values("metric", ignore_index=True)

    if len(images) != expected_images:
        raise AssertionError(
            f"Expected {expected_images} images, found {len(images)}"
        )
    if images["imagem_id"].duplicated().any():
        raise AssertionError("Duplicate imagem_id values in WIDER inventory")
    known_ids = set(images["imagem_id"].astype(str))
    for label, table in (("truth", truth), ("predictions", predictions)):
        unknown = set(table["imagem_id"].astype(str)) - known_ids
        if unknown:
            raise AssertionError(f"{label} contains unknown imagem_id values")

    per_image, summary = evaluate_detector(predictions, truth, images)
    ci = bootstrap_detector(per_image).sort_values("metric", ignore_index=True)
    per_image = per_image.sort_values("imagem_id", ignore_index=True)
    summary["ap50_global"] = average_precision_at_iou(predictions, truth)
    summary = summary.sort_values("grupo", ignore_index=True)
    pd.testing.assert_frame_equal(
        saved_per_image,
        per_image,
        check_dtype=False,
        atol=1e-12,
        rtol=1e-12,
    )
    pd.testing.assert_frame_equal(
        saved_summary,
        summary,
        check_dtype=False,
        atol=1e-12,
        rtol=1e-12,
    )
    pd.testing.assert_frame_equal(
        saved_ci,
        ci,
        check_dtype=False,
        atol=1e-12,
        rtol=1e-12,
    )

    if metadata.get("image_count") != expected_images:
        raise AssertionError("Metadata image_count does not match the inventory")
    overall = summary.loc[summary["grupo"] == "overall"].iloc[0]
    if not np.isclose(overall["ap50_global"], average_precision_at_iou(predictions, truth)):
        raise AssertionError("AP50 mismatch")
    return {
        "images": len(images),
        "ground_truth_boxes": len(truth),
        "predictions": len(predictions),
        "precision": overall["precision"],
        "recall": overall["recall"],
        "f1": overall["f1"],
        "mean_iou": overall["mean_iou"],
        "ap50": overall["ap50_global"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-images", type=int)
    args = parser.parse_args()

    metadata = _load_metadata(args.output_dir)
    mode = metadata.get("validation_mode")
    expected_images = args.expected_images or (
        3226 if mode == "wider-face-benchmark" else 64
    )
    if mode == "wider-face-benchmark":
        result = _validate_wider(args.output_dir, expected_images, metadata)
    elif mode == "synthetic-detector-disagreement":
        result = _validate_synthetic(args.output_dir, expected_images, metadata)
    else:
        raise ValueError(f"Unknown detector validation mode: {mode!r}")
    print(
        json.dumps(
            result,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
