"""Recursive, structure-independent dataset discovery."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError

from .config import DatasetConfig


MANIFEST_COLUMNS = {
    "relative_path",
    "source_image_id",
    "modelo_ia",
    "tipo_prompt",
    "prompt_id",
    "pair_id",
    "analysis_eligible",
}


@dataclass(frozen=True)
class DatasetDiscovery:
    """Validated image inventory and complete ingestion report."""

    images: pd.DataFrame
    report: pd.DataFrame


def normalize_relative_path(path: Path | str) -> str:
    """Normalize a relative path for deterministic comparisons."""

    value = Path(path).as_posix()
    return unicodedata.normalize("NFC", value).casefold().lstrip("./")


def content_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest without loading the complete file in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def make_image_id(dataset_id: str, digest: str) -> str:
    """Create a stable identifier independent of directory depth."""

    value = f"{dataset_id.strip().casefold()}\0{digest}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:16]


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _read_manifest(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=sorted(MANIFEST_COLUMNS))
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    manifest = pd.read_csv(path, dtype="string")
    if "relative_path" not in manifest:
        raise ValueError("Manifest must contain relative_path")
    manifest["relative_path_key"] = manifest["relative_path"].map(normalize_relative_path)
    if manifest["relative_path_key"].duplicated().any():
        duplicates = manifest.loc[
            manifest["relative_path_key"].duplicated(keep=False), "relative_path"
        ].tolist()
        raise ValueError(f"Manifest contains duplicate relative_path values: {duplicates}")
    return manifest


def _verify_image(path: Path) -> tuple[bool, str | None]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.convert("RGB").load()
        return True, None
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().casefold()
    if text in {"1", "true", "yes", "sim"}:
        return True
    if text in {"0", "false", "no", "nao"}:
        return False
    raise ValueError(f"Invalid analysis_eligible value: {value}")


def discover_dataset(config: DatasetConfig) -> DatasetDiscovery:
    """Recursively discover valid images without interpreting folder positions."""

    root = config.input_root.resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    output = config.output_dir.resolve()
    manifest = _read_manifest(config.manifest_path)
    manifest_map = (
        manifest.set_index("relative_path_key").to_dict(orient="index")
        if not manifest.empty
        else {}
    )

    candidates: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        if _inside(path, output):
            continue
        if path.suffix.casefold() in config.image_extensions:
            candidates.append(path)
    candidates.sort(key=lambda path: normalize_relative_path(path.relative_to(root)))
    if not candidates:
        raise FileNotFoundError(f"No supported images found in {root}")

    image_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    seen_hashes: dict[str, str] = {}
    discovered_keys: set[str] = set()

    for path in candidates:
        relative = path.relative_to(root)
        key = normalize_relative_path(relative)
        discovered_keys.add(key)
        valid, error = _verify_image(path)
        report = {
            "caminho_relativo": relative.as_posix(),
            "status_ingestao": "valid" if valid else "invalid",
            "erro_ingestao": error,
            "duplicate_of": pd.NA,
        }
        if not valid:
            report_rows.append(report)
            continue

        digest = content_sha256(path)
        if digest in seen_hashes:
            report["status_ingestao"] = "duplicate"
            report["duplicate_of"] = seen_hashes[digest]
            report_rows.append(report)
            if config.duplicate_policy == "error":
                raise ValueError(
                    "Duplicate image content detected: "
                    f"{seen_hashes[digest]} and {relative.as_posix()}"
                )
            continue

        seen_hashes[digest] = relative.as_posix()
        metadata = manifest_map.get(key, {})
        image_id = make_image_id(config.dataset_id, digest)
        image_rows.append(
            {
                "dataset_id": config.dataset_id,
                "imagem_id": image_id,
                "content_sha256": digest,
                "caminho_relativo": relative.as_posix(),
                "nome_arquivo": path.name,
                "source_image_id": metadata.get("source_image_id", pd.NA),
                "prompt_id": metadata.get("prompt_id", pd.NA),
                "pair_id": metadata.get("pair_id", pd.NA),
                "tipo_prompt": metadata.get("tipo_prompt", pd.NA),
                "modelo_ia": metadata.get("modelo_ia", pd.NA),
                "analysis_eligible": _coerce_optional_bool(
                    metadata.get("analysis_eligible")
                ),
                "absolute_path": path,
            }
        )
        report_rows.append(report)

    manifest_only = sorted(set(manifest_map) - discovered_keys)
    if manifest_only:
        raise ValueError(f"Manifest references missing images: {manifest_only}")

    images = pd.DataFrame(image_rows)
    report = pd.DataFrame(report_rows)
    invalid_count = int((report["status_ingestao"] == "invalid").sum())
    if config.strict_ingestion and invalid_count:
        raise ValueError(f"Ingestion found {invalid_count} invalid image(s)")
    if images.empty:
        raise ValueError("No valid unique images remain after ingestion")
    if config.expected_images is not None and len(images) != config.expected_images:
        raise ValueError(
            f"Expected {config.expected_images} unique images; found {len(images)}"
        )
    return DatasetDiscovery(images=images, report=report)
