from __future__ import annotations

from pathlib import Path

from PIL import Image


def make_image(path: Path, color: tuple[int, int, int] = (80, 120, 160)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 18), color).save(path)
    return path

