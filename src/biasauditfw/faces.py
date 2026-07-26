"""DeepFace 1:N normalization and adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _confidence(result: dict[str, Any], dominant_key: str, scores_key: str) -> float | None:
    dominant = result.get(dominant_key)
    scores = result.get(scores_key)
    if not dominant or not isinstance(scores, dict):
        return None
    value = scores.get(dominant)
    return float(value) if value is not None else None


def normalize_deepface_results(raw: Any) -> list[dict[str, Any]]:
    """Normalize zero, one or many DeepFace results and sort spatially."""

    if raw is None:
        return []
    values = raw if isinstance(raw, list) else [raw]
    faces: list[dict[str, Any]] = []
    for result in values:
        if not isinstance(result, dict):
            continue
        region = result.get("region") or {}
        x = max(0, int(region.get("x", 0) or 0))
        y = max(0, int(region.get("y", 0) or 0))
        w = int(region.get("w", 0) or 0)
        h = int(region.get("h", 0) or 0)
        if w <= 0 or h <= 0:
            continue
        faces.append(
            {
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "raca": result.get("dominant_race"),
                "confianca_raca": _confidence(
                    result, "dominant_race", "race"
                ),
                "genero": result.get("dominant_gender"),
                "confianca_genero": _confidence(
                    result, "dominant_gender", "gender"
                ),
            }
        )
    return sorted(
        faces,
        key=lambda face: (
            face["bbox_y"],
            face["bbox_x"],
            face["bbox_w"],
            face["bbox_h"],
        ),
    )


def analyze_faces(
    image_path: Path, detector_backend: str = "retinaface"
) -> tuple[list[dict[str, Any]], str, str | None]:
    """Run DeepFace while preserving images with no valid detections."""

    from deepface import DeepFace

    try:
        raw = DeepFace.analyze(
            img_path=str(image_path),
            actions=["gender", "race"],
            detector_backend=detector_backend,
            enforce_detection=True,
            silent=True,
        )
        faces = normalize_deepface_results(raw)
        return faces, ("detected" if faces else "no_face"), None
    except ValueError as exc:
        message = str(exc)
        if "face" in message.casefold() and "detect" in message.casefold():
            return [], "no_face", None
        return [], "error", f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        return [], "error", f"{type(exc).__name__}: {exc}"
