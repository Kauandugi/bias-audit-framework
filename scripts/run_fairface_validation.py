"""Compare DeepFace labels with FairFace on the same detected face crops."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as package_metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from biasauditfw.neural import (
    FairFaceOracle,
    download_fairface_weights,
    run_fairface_on_detected_faces,
)
from biasauditfw.validation import concordance_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-csv", type=Path, required=True)
    parser.add_argument("--faces-csv", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        download_fairface_weights(args.weights)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device_name = (
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    device = torch.device(device_name)
    images = pd.read_csv(args.images_csv)
    faces = pd.read_csv(args.faces_csv)
    oracle = FairFaceOracle(args.weights, device)
    comparison = run_fairface_on_detected_faces(
        oracle,
        faces,
        images,
        args.input_root,
    )
    concordance, matrices = concordance_report(comparison)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(
        args.output_dir / "deepface_fairface_comparison.csv",
        index=False,
    )
    concordance.to_csv(
        args.output_dir / "deepface_fairface_concordance.csv",
        index=False,
    )
    for attribute, matrix in matrices.items():
        matrix.to_csv(args.output_dir / f"confusion_{attribute}.csv")
    metadata = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": device_name,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "torchvision_version": package_metadata.version("torchvision"),
        "weights_path": str(args.weights),
        "weights_sha256": file_sha256(args.weights),
        "images": int(images["imagem_id"].nunique()),
        "faces": len(faces),
        "comparison_rows": len(comparison),
        "interpretation": (
            "FairFace is a secondary pseudo-reference on RetinaFace crops, "
            "not demographic Ground Truth."
        ),
    }
    (args.output_dir / "fairface_run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(concordance.to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
