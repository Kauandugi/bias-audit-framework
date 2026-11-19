"""Manifest builders for datasets distributed with per-image sidecar metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_EXTENSIONS
from .ingestion import is_image_candidate


def _required_text(metadata: dict[str, Any], field: str, source: Path) -> str:
    value = str(metadata.get(field, "")).strip()
    if not value:
        raise ValueError(f"{source} is missing a non-empty {field}")
    return value


def _prompt_group(value: str, source: Path) -> tuple[str, str]:
    normalized = value.strip().casefold()
    if normalized in {"neutral", "neutro"}:
        return "Neutral Prompts", "N"
    if normalized in {"inclusive", "inclusivo", "inclusiva"}:
        return "Inclusive Prompts", "I"
    raise ValueError(f"{source} contains an unknown prompt_type: {value}")


def build_sidecar_manifest(
    input_root: Path,
    dataset_id: str,
    expected_images: int | None = None,
    require_complete_pairs: bool = True,
) -> pd.DataFrame:
    """Build a schema 2.0 manifest from adjacent ``metadata.json`` files.

    Each supported image must have a ``metadata.json`` in the same directory
    containing ``prompt_id``, ``prompt_type`` and ``model``. Pair identifiers
    include the dataset and model so that equal prompt IDs from different
    generators cannot collide.
    """

    root = Path(input_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    dataset = dataset_id.strip()
    if not dataset:
        raise ValueError("dataset_id must not be empty")

    extensions = {suffix.casefold() for suffix in DEFAULT_EXTENSIONS}
    images = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and is_image_candidate(path, tuple(extensions))
        ),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )
    if not images:
        raise FileNotFoundError(f"No supported images found in {root}")
    if expected_images is not None and len(images) != expected_images:
        raise ValueError(f"Expected {expected_images} images; found {len(images)}")

    records: list[dict[str, Any]] = []
    for image_path in images:
        metadata_path = image_path.with_name("metadata.json")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing sidecar metadata: {metadata_path}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {metadata_path}: {exc}") from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"{metadata_path} must contain a JSON object")

        prompt_id = _required_text(metadata, "prompt_id", metadata_path).upper()
        model = _required_text(metadata, "model", metadata_path)
        prompt_type, suffix = _prompt_group(
            _required_text(metadata, "prompt_type", metadata_path), metadata_path
        )
        pair_id = f"{dataset}:{model}:{prompt_id}"
        records.append(
            {
                "relative_path": image_path.relative_to(root).as_posix(),
                "source_image_id": f"{pair_id}:{suffix}",
                "modelo_ia": model,
                "tipo_prompt": prompt_type,
                "prompt_id": prompt_id,
                "pair_id": pair_id,
                "analysis_eligible": True,
            }
        )

    manifest = pd.DataFrame(records).sort_values(
        ["modelo_ia", "prompt_id", "tipo_prompt"], ignore_index=True
    )
    if not manifest["relative_path"].is_unique:
        raise ValueError("relative_path must be unique")
    if not manifest["source_image_id"].is_unique:
        raise ValueError("source_image_id must be unique")
    if manifest.duplicated(["pair_id", "tipo_prompt"]).any():
        raise ValueError("Each pair may contain at most one image per prompt group")
    if require_complete_pairs:
        pair_groups = manifest.groupby("pair_id")["tipo_prompt"].agg(set)
        expected = {"Neutral Prompts", "Inclusive Prompts"}
        incomplete = pair_groups[pair_groups.map(lambda groups: groups != expected)]
        if not incomplete.empty:
            raise ValueError(
                "Every pair_id must contain one neutral and one inclusive image: "
                f"{incomplete.index.tolist()}"
            )
    return manifest


def build_path_manifest(
    input_root: Path,
    dataset_id: str,
    model_names: dict[str, str] | None = None,
    expected_images: int | None = None,
) -> pd.DataFrame:
    """Build an explicit manifest for a known model/group/prompt folder layout.

    This migration helper is intentionally separate from dataset ingestion. It
    interprets the first path segment as a model label and recognizes neutral
    or inclusive group names plus ``prompt N`` anywhere in the relative path.
    Incomplete pairs remain in the manifest but are excluded from inference.
    """

    root = Path(input_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    dataset = dataset_id.strip()
    if not dataset:
        raise ValueError("dataset_id must not be empty")
    aliases = model_names or {}
    extensions = {suffix.casefold() for suffix in DEFAULT_EXTENSIONS}
    images = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and is_image_candidate(path, tuple(extensions))
        ),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )
    if not images:
        raise FileNotFoundError(f"No supported images found in {root}")
    if expected_images is not None and len(images) != expected_images:
        raise ValueError(f"Expected {expected_images} images; found {len(images)}")

    records: list[dict[str, Any]] = []
    for image_path in images:
        relative = image_path.relative_to(root)
        relative_text = relative.as_posix()
        if not relative.parts:
            raise ValueError(f"Cannot infer model from {relative_text}")
        model_key = relative.parts[0]
        model = aliases.get(model_key, model_key).strip()
        folded = relative_text.casefold()
        if "inclus" in folded:
            prompt_type, suffix = "Inclusive Prompts", "I"
        elif "neutral" in folded or "neutro" in folded:
            prompt_type, suffix = "Neutral Prompts", "N"
        else:
            raise ValueError(f"Cannot infer prompt group from {relative_text}")
        match = re.search(r"prompt\s*(\d+)", relative_text, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"Cannot infer prompt_id from {relative_text}")
        prompt_id = f"P{int(match.group(1))}"
        pair_id = f"{dataset}:{model}:{prompt_id}"
        records.append(
            {
                "relative_path": relative_text,
                "source_image_id": f"{pair_id}:{suffix}",
                "modelo_ia": model,
                "tipo_prompt": prompt_type,
                "prompt_id": prompt_id,
                "pair_id": pair_id,
                "analysis_eligible": False,
            }
        )

    manifest = pd.DataFrame(records)
    if manifest.duplicated(["pair_id", "tipo_prompt"]).any():
        raise ValueError("Each pair may contain at most one image per prompt group")
    groups = manifest.groupby("pair_id")["tipo_prompt"].agg(set)
    complete = groups[groups.map(
        lambda values: values == {"Neutral Prompts", "Inclusive Prompts"}
    )].index
    manifest["analysis_eligible"] = manifest["pair_id"].isin(complete)
    return manifest.sort_values(
        ["modelo_ia", "prompt_id", "tipo_prompt"], ignore_index=True
    )
