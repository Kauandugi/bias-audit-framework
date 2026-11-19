"""Run WIDER FACE benchmark or synthetic detector-disagreement validation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from biasauditfw.faces import detect_face_boxes
from biasauditfw.validation import (
    average_precision_at_iou,
    bootstrap_detector,
    bootstrap_detector_disagreement,
    compare_detectors,
    evaluate_detector,
    wider_face_to_dataframes,
)


def write_run_metadata(
    output_dir: Path,
    *,
    mode: str,
    image_count: int,
    input_description: dict[str, str],
) -> None:
    """Persist enough provenance to reproduce a validation run."""

    package_names = ("biasauditfw", "deepface", "mtcnn", "numpy", "pandas")
    versions: dict[str, str] = {}
    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = "not-installed"
    payload = {
        "validation_mode": mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_count": image_count,
        "iou_threshold": 0.5,
        "python": platform.python_version(),
        "packages": versions,
        "inputs": input_description,
    }
    (output_dir / "detector_validation_metadata.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def detect(images: pd.DataFrame, input_root: Path, backend: str) -> pd.DataFrame:
    rows = []
    for image in images.itertuples(index=False):
        boxes = detect_face_boxes(input_root / image.caminho_relativo, backend)
        if boxes.empty:
            continue
        boxes.insert(0, "imagem_id", image.imagem_id)
        rows.extend(boxes.to_dict(orient="records"))
    return pd.DataFrame(
        rows,
        columns=[
            "imagem_id", "detector", "detector_confidence", "bbox_x",
            "bbox_y", "bbox_w", "bbox_h",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    wider = subparsers.add_parser("wider")
    wider.add_argument("--annotations", type=Path, required=True)
    wider.add_argument("--images-root", type=Path, required=True)
    wider.add_argument("--output-dir", type=Path, required=True)
    synthetic = subparsers.add_parser("synthetic")
    synthetic.add_argument("--images-csv", type=Path, required=True)
    synthetic.add_argument("--input-root", type=Path, required=True)
    synthetic.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "wider":
        images, truth = wider_face_to_dataframes(args.annotations)
        predictions = detect(images, args.images_root, "retinaface")
        per_image, summary = evaluate_detector(predictions, truth, images)
        confidence_intervals = bootstrap_detector(per_image)
        ap50 = average_precision_at_iou(predictions, truth)
        summary["ap50_global"] = ap50
        images.to_csv(args.output_dir / "wider_images.csv", index=False)
        truth.to_csv(args.output_dir / "wider_ground_truth.csv", index=False)
        predictions.to_csv(args.output_dir / "wider_retinaface_predictions.csv", index=False)
        per_image.to_csv(args.output_dir / "wider_detector_per_image.csv", index=False)
        summary.to_csv(args.output_dir / "wider_detector_summary.csv", index=False)
        confidence_intervals.to_csv(
            args.output_dir / "wider_detector_bootstrap_ci.csv", index=False
        )
        write_run_metadata(
            args.output_dir,
            mode="wider-face-benchmark",
            image_count=len(images),
            input_description={
                "annotations": str(args.annotations.resolve()),
                "images_root": str(args.images_root.resolve()),
                "benchmark_role": (
                    "External benchmark consistency test; not synthetic-domain "
                    "Ground Truth"
                ),
            },
        )
        return

    images = pd.read_csv(args.images_csv)
    retinaface = detect(images, args.input_root, "retinaface")
    mtcnn = detect(images, args.input_root, "mtcnn")
    per_image, summary = compare_detectors(retinaface, mtcnn, images)
    confidence_intervals = bootstrap_detector_disagreement(per_image)
    retinaface.to_csv(args.output_dir / "synthetic_retinaface_boxes.csv", index=False)
    mtcnn.to_csv(args.output_dir / "synthetic_mtcnn_boxes.csv", index=False)
    per_image.to_csv(args.output_dir / "detector_disagreement_per_image.csv", index=False)
    summary.to_csv(args.output_dir / "detector_disagreement_summary.csv", index=False)
    confidence_intervals.to_csv(
        args.output_dir / "detector_disagreement_bootstrap_ci.csv", index=False
    )
    write_run_metadata(
        args.output_dir,
        mode="synthetic-detector-disagreement",
        image_count=len(images),
        input_description={
            "images_csv": str(args.images_csv.resolve()),
            "input_root": str(args.input_root.resolve()),
            "reference_role": "MTCNN is a pseudo-reference, not Ground Truth",
        },
    )


if __name__ == "__main__":
    main()
