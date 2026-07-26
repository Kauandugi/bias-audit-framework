"""Ground Truth, detector and pseudo-oracle validation utilities."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from .ingestion import normalize_relative_path


BOX_COLUMNS = ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]


def coco_to_dataframe(
    coco_path: Path, images: pd.DataFrame, annotator: str
) -> pd.DataFrame:
    """Map COCO annotations by exact relative path or explicit source ID."""

    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    image_map = {item["id"]: item for item in coco.get("images", [])}
    relative_map = {
        normalize_relative_path(path): image_id
        for path, image_id in zip(images["caminho_relativo"], images["imagem_id"])
    }
    source_map = {
        str(source): image_id
        for source, image_id in zip(images["source_image_id"], images["imagem_id"])
        if pd.notna(source)
    }
    rows = []
    counters: dict[str, int] = {}
    for annotation in coco.get("annotations", []):
        item = image_map[annotation["image_id"]]
        source_id = item.get("source_image_id")
        relative_key = normalize_relative_path(item["file_name"])
        image_id = source_map.get(str(source_id)) if source_id is not None else None
        image_id = image_id or relative_map.get(relative_key)
        if image_id is None:
            raise ValueError(
                f"COCO image is absent from inventory: {item['file_name']}"
            )
        counters[image_id] = counters.get(image_id, 0) + 1
        x, y, w, h = map(float, annotation["bbox"])
        rows.append(
            {
                "imagem_id": image_id,
                "gt_face_id": f"{image_id}_gt{counters[image_id]:03d}",
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "avaliador": annotator,
            }
        )
    return pd.DataFrame(rows)


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax, ay, aw, ah = map(float, box_a)
    bx, by, bw, bh = map(float, box_b)
    x_left, y_top = max(ax, bx), max(ay, by)
    x_right, y_bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, x_right - x_left) * max(0.0, y_bottom - y_top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(
    predicted: pd.DataFrame, truth: pd.DataFrame, threshold: float = 0.5
) -> list[tuple[int, int, float]]:
    if predicted.empty or truth.empty:
        return []
    ious = np.array(
        [
            [bbox_iou(prediction, target) for target in truth[BOX_COLUMNS].to_numpy()]
            for prediction in predicted[BOX_COLUMNS].to_numpy()
        ]
    )
    predicted_indices, truth_indices = linear_sum_assignment(1.0 - ious)
    return [
        (int(prediction), int(target), float(ious[prediction, target]))
        for prediction, target in zip(predicted_indices, truth_indices)
        if ious[prediction, target] >= threshold
    ]


def evaluate_detector(
    faces: pd.DataFrame, truth: pd.DataFrame, images: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for image_id in images["imagem_id"]:
        predicted = faces[faces["imagem_id"] == image_id].reset_index(drop=True)
        target = truth[truth["imagem_id"] == image_id].reset_index(drop=True)
        matches = match_boxes(predicted, target)
        rows.append(
            {
                "imagem_id": image_id,
                "tp": len(matches),
                "fp": len(predicted) - len(matches),
                "fn": len(target) - len(matches),
                "sum_iou": sum(match[2] for match in matches),
            }
        )
    per_image = pd.DataFrame(rows).merge(
        images[["imagem_id", "modelo_ia", "tipo_prompt"]], on="imagem_id"
    )

    def summarize(group: pd.DataFrame, label: str) -> dict[str, Any]:
        tp, fp, fn = group[["tp", "fp", "fn"]].sum()
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else np.nan
        )
        return {
            "grupo": label,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou": group["sum_iou"].sum() / tp if tp else np.nan,
        }

    summaries = [summarize(per_image, "overall")]
    summaries.extend(
        summarize(group, f"modelo={name}")
        for name, group in per_image.groupby("modelo_ia", dropna=False)
    )
    summaries.extend(
        summarize(group, f"prompt={name}")
        for name, group in per_image.groupby("tipo_prompt", dropna=False)
    )
    return per_image, pd.DataFrame(summaries)


def bootstrap_detector(
    per_image: pd.DataFrame, iterations: int = 2000, seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metrics = []
    for _ in range(iterations):
        sample = per_image.iloc[rng.integers(0, len(per_image), len(per_image))]
        tp, fp, fn = sample[["tp", "fp", "fn"]].sum()
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else np.nan
        )
        metrics.append((precision, recall, f1))
    values = np.asarray(metrics, dtype=float)
    return pd.DataFrame(
        {
            "metric": ["precision", "recall", "f1"],
            "ci_2_5": np.nanpercentile(values, 2.5, axis=0),
            "ci_97_5": np.nanpercentile(values, 97.5, axis=0),
        }
    )


def harmonize_race(value: Any) -> str:
    text = str(value).casefold().replace("_", " ")
    if text in {"east asian", "southeast asian", "asian"}:
        return "asian"
    return {
        "white": "white",
        "black": "black",
        "latino hispanic": "latino hispanic",
        "indian": "indian",
        "middle eastern": "middle eastern",
    }.get(text, "unmapped")


def harmonize_gender(value: Any) -> str:
    text = str(value).casefold()
    if text in {"man", "male"}:
        return "male"
    if text in {"woman", "female"}:
        return "female"
    return "unmapped"


def concordance_report(
    merged: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compare aligned DeepFace/FairFace labels without claiming identity accuracy."""

    data = merged.copy()
    data["deepface_race_h"] = data["deepface_raca"].map(harmonize_race)
    data["fairface_race_h"] = data["fairface_raca"].map(harmonize_race)
    data["deepface_gender_h"] = data["deepface_genero"].map(harmonize_gender)
    data["fairface_gender_h"] = data["fairface_genero"].map(harmonize_gender)
    reports, matrices = [], {}
    for attribute in ("race", "gender"):
        deep = data[f"deepface_{attribute}_h"]
        fair = data[f"fairface_{attribute}_h"]
        valid = (deep != "unmapped") & (fair != "unmapped")
        labels = sorted(set(deep[valid]) | set(fair[valid]))
        if not labels:
            reports.append(
                {
                    "attribute": attribute,
                    "n": 0,
                    "agreement": np.nan,
                    "cohen_kappa": np.nan,
                    "macro_f1_pseudo_reference": np.nan,
                }
            )
            matrices[attribute] = pd.DataFrame()
            continue
        reports.append(
            {
                "attribute": attribute,
                "n": int(valid.sum()),
                "agreement": float((deep[valid] == fair[valid]).mean()),
                "cohen_kappa": cohen_kappa_score(fair[valid], deep[valid]),
                "macro_f1_pseudo_reference": f1_score(
                    fair[valid], deep[valid], average="macro"
                ),
            }
        )
        matrices[attribute] = pd.DataFrame(
            confusion_matrix(fair[valid], deep[valid], labels=labels),
            index=labels,
            columns=labels,
        )
    return pd.DataFrame(reports), matrices

