"""Configuration types shared by local tests and the Colab notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
)


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset discovery and output configuration."""

    input_root: Path
    output_dir: Path
    dataset_id: str
    manifest_path: Path | None = None
    expected_images: int | None = None
    image_extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    duplicate_policy: str = "error"
    strict_ingestion: bool = True
    detector_backend: str = "retinaface"
    clip_model_id: str = "openai/clip-vit-base-patch32"
    seed: int = 42
    smoke_images: int = 4

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("dataset_id must not be empty")
        if self.duplicate_policy not in {"error", "first"}:
            raise ValueError("duplicate_policy must be 'error' or 'first'")
        normalized = tuple(
            suffix.casefold() if suffix.startswith(".") else f".{suffix.casefold()}"
            for suffix in self.image_extensions
        )
        object.__setattr__(self, "image_extensions", normalized)

