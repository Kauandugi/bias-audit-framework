"""Run the canonical schema 2.0 pipeline for one explicit dataset."""

from __future__ import annotations

import argparse
import importlib.metadata as package_metadata
import json
import platform
from functools import partial
from pathlib import Path

import torch

from biasauditfw import (
    DatasetConfig,
    discover_dataset,
    export_results,
    process_dataset,
    select_smoke_inventory,
)
from biasauditfw.clip_diversity import DEFAULT_PROTOTYPE_TEXTS, ClipDiversityScorer
from biasauditfw.config import DEFAULT_CLIP_MODEL_REVISION
from biasauditfw.faces import analyze_faces
from biasauditfw.reporting import export_statistical_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--expected-images", type=int)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-model-revision", default=DEFAULT_CLIP_MODEL_REVISION)
    parser.add_argument("--detector-backend", default="retinaface")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-processing-errors",
        action="store_true",
        help="Export diagnostic rows instead of failing when a neural stage errors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DatasetConfig(
        input_root=args.input_root,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        manifest_path=args.manifest,
        expected_images=args.expected_images,
        detector_backend=args.detector_backend,
        clip_model_id=args.clip_model_id,
        clip_model_revision=args.clip_model_revision,
        seed=args.seed,
    )
    discovery = discover_dataset(config)
    inventory = discovery.images
    run_inventory = (
        select_smoke_inventory(inventory, args.max_images)
        if args.max_images is not None
        else inventory
    )
    print(
        json.dumps(
            {
                "dataset_id": args.dataset_id,
                "discovered_images": len(inventory),
                "selected_images": len(run_inventory),
                "device": str(device),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    scorer = ClipDiversityScorer(
        config.clip_model_id,
        device,
        revision=config.clip_model_revision,
    )
    face_analyzer = partial(
        analyze_faces,
        detector_backend=config.detector_backend,
    )
    images, faces = process_dataset(
        run_inventory,
        face_analyzer=face_analyzer,
        semantic_scorer=scorer,
    )
    run_mode = "full" if len(run_inventory) == len(inventory) else "smoke"
    metadata = {
        "dataset_id": args.dataset_id,
        "run_mode": run_mode,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": package_metadata.version("transformers"),
        "deepface_version": package_metadata.version("deepface"),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "clip_model_id": config.clip_model_id,
        "clip_model_revision": config.clip_model_revision,
        "clip_scoring": "L2-normalized cosine prototype margins",
        "clip_prototype_texts": DEFAULT_PROTOTYPE_TEXTS,
        "detector_backend": config.detector_backend,
        "seed": config.seed,
        "manifest_path": str(args.manifest),
    }
    export_results(
        images,
        faces,
        discovery.report,
        args.output_dir,
        metadata,
        expected_images=len(run_inventory),
    )
    inference = export_statistical_reports(images, faces, args.output_dir)
    detection_errors = int(images["status_deteccao"].eq("error").sum())
    clip_errors = int(images["status_clip"].eq("error").sum())
    print(
        json.dumps(
            {
                "images": len(images),
                "faces": len(faces),
                "detection_errors": detection_errors,
                "clip_errors": clip_errors,
                "inference": inference,
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not args.allow_processing_errors and (detection_errors or clip_errors):
        raise RuntimeError(
            "Neural processing failed for "
            f"{detection_errors} detection rows and {clip_errors} CLIP rows. "
            "Diagnostic artifacts were preserved in the output directory."
        )


if __name__ == "__main__":
    main()
