"""Image-level and face-level orchestration."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import CLIP_OUTPUT_COLUMNS
from .contracts import FACE_COLUMNS, IMAGE_COLUMNS, SCHEMA_VERSION, validate_contract


FaceAnalyzer = Callable[[Path], tuple[list[dict[str, Any]], str, str | None]]
SemanticScorer = Callable[[Path], dict[str, float]]


def process_dataset(
    inventory: pd.DataFrame,
    face_analyzer: FaceAnalyzer,
    semantic_scorer: SemanticScorer,
    max_images: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process validated inventory without inferring metadata from directories."""

    records = inventory.iloc[:max_images] if max_images is not None else inventory
    image_rows: list[dict[str, Any]] = []
    face_rows: list[dict[str, Any]] = []
    metadata_columns = [
        column
        for column in IMAGE_COLUMNS
        if column
        not in {
            "num_faces",
            "status_deteccao",
            "erro_deteccao",
            "status_clip",
            "erro_clip",
            *CLIP_OUTPUT_COLUMNS,
        }
    ]

    for record in records.to_dict(orient="records"):
        path = Path(record["absolute_path"])
        faces, detection_status, detection_error = face_analyzer(path)
        try:
            clip_scores = semantic_scorer(path)
            missing = set(CLIP_OUTPUT_COLUMNS) - set(clip_scores)
            if missing:
                raise ValueError(f"Semantic scorer omitted columns: {sorted(missing)}")
            clip_status, clip_error = "ok", None
        except Exception as exc:
            clip_scores = {column: float("nan") for column in CLIP_OUTPUT_COLUMNS}
            clip_status = "error"
            clip_error = f"{type(exc).__name__}: {exc}"

        image_meta = {column: record.get(column, pd.NA) for column in metadata_columns}
        image_rows.append(
            {
                **image_meta,
                "num_faces": len(faces),
                "status_deteccao": detection_status,
                "erro_deteccao": detection_error,
                "status_clip": clip_status,
                "erro_clip": clip_error,
                **clip_scores,
            }
        )
        for index, face in enumerate(faces, start=1):
            face_rows.append(
                {
                    "dataset_id": record["dataset_id"],
                    "imagem_id": record["imagem_id"],
                    "face_id": f"{record['imagem_id']}_f{index:03d}",
                    "num_faces": len(faces),
                    **face,
                }
            )

    return (
        pd.DataFrame(image_rows, columns=IMAGE_COLUMNS),
        pd.DataFrame(face_rows, columns=FACE_COLUMNS),
    )


def export_results(
    images: pd.DataFrame,
    faces: pd.DataFrame,
    ingestion_report: pd.DataFrame,
    output_dir: Path,
    metadata: dict[str, Any],
    expected_images: int | None = None,
) -> None:
    """Write canonical artifacts without overwriting legacy outputs implicitly."""

    validate_contract(images, faces, expected_images)
    output_dir.mkdir(parents=True, exist_ok=True)
    images.to_csv(output_dir / "audit_images.csv", index=False, encoding="utf-8-sig")
    faces.to_csv(output_dir / "audit_faces.csv", index=False, encoding="utf-8-sig")
    ingestion_report.to_csv(
        output_dir / "ingestion_report.csv", index=False, encoding="utf-8-sig"
    )
    run_metadata = {
        "schema_version": SCHEMA_VERSION,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **metadata,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
