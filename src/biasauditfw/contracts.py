"""Canonical schema 2.0 contracts and invariant checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import CLIP_OUTPUT_COLUMNS


SCHEMA_VERSION = "2.0"

IMAGE_COLUMNS = [
    "dataset_id",
    "imagem_id",
    "content_sha256",
    "caminho_relativo",
    "nome_arquivo",
    "source_image_id",
    "prompt_id",
    "pair_id",
    "tipo_prompt",
    "modelo_ia",
    "analysis_eligible",
    "num_faces",
    "status_deteccao",
    "erro_deteccao",
    "status_clip",
    "erro_clip",
    *CLIP_OUTPUT_COLUMNS,
]

FACE_COLUMNS = [
    "dataset_id",
    "imagem_id",
    "face_id",
    "num_faces",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "raca",
    "confianca_raca",
    "genero",
    "confianca_genero",
]

OLD_CLIP_COLUMNS = {
    "clip_latin_american",
    "clip_not_latin_american",
    "clip_margin",
}


def validate_contract(
    images: pd.DataFrame,
    faces: pd.DataFrame,
    expected_images: int | None = None,
) -> None:
    """Validate units, referential integrity and numerical ranges."""

    if list(images.columns) != IMAGE_COLUMNS:
        raise ValueError("audit_images schema does not match schema 2.0")
    if list(faces.columns) != FACE_COLUMNS:
        raise ValueError("audit_faces schema does not match schema 2.0")
    if OLD_CLIP_COLUMNS & set(images):
        raise ValueError("Legacy Latin American CLIP columns are forbidden")
    if any(column.startswith("clip_") for column in faces):
        raise ValueError("Face table must not contain CLIP columns")
    if images["imagem_id"].isna().any() or not images["imagem_id"].is_unique:
        raise ValueError("imagem_id must be present and unique")
    if images["content_sha256"].isna().any() or not images["content_sha256"].is_unique:
        raise ValueError("content_sha256 must be present and unique")
    if expected_images is not None and len(images) != expected_images:
        raise ValueError(f"Expected {expected_images} images; found {len(images)}")
    if faces["face_id"].isna().any() or not faces["face_id"].is_unique:
        raise ValueError("face_id must be present and unique")
    if not set(faces["imagem_id"]).issubset(set(images["imagem_id"])):
        raise ValueError("Face table contains an unknown imagem_id")
    expected = images.set_index("imagem_id")["num_faces"].astype(int)
    observed = faces.groupby("imagem_id").size().reindex(expected.index, fill_value=0)
    if not observed.equals(expected):
        raise ValueError("num_faces does not match face rows")
    if not faces.empty:
        if not (faces[["bbox_x", "bbox_y"]] >= 0).all().all():
            raise ValueError("Face boxes contain a negative origin")
        if not (faces[["bbox_w", "bbox_h"]] > 0).all().all():
            raise ValueError("Face boxes contain non-positive dimensions")
    for column in CLIP_OUTPUT_COLUMNS:
        valid = images[column].dropna()
        low, high = (-2.0, 2.0) if column.endswith("_margin") else (-1.0, 1.0)
        if not valid.between(low - 1e-6, high + 1e-6).all():
            raise ValueError(f"{column} is outside [{low}, {high}]")
    for prefix in ("racial", "gender"):
        expected_margin = (
            images[f"clip_{prefix}_diversity_cosine"]
            - images[f"clip_{prefix}_homogeneity_cosine"]
        )
        actual = images[f"clip_{prefix}_diversity_margin"]
        valid = expected_margin.notna() & actual.notna()
        if not np.allclose(expected_margin[valid], actual[valid], atol=1e-7):
            raise ValueError(f"{prefix} CLIP margin is inconsistent")
