"""Ground Truth, detector and pseudo-oracle validation utilities."""

from __future__ import annotations

import json
import hashlib
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from .ingestion import normalize_relative_path


BOX_COLUMNS = ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]


def wider_face_to_dataframes(
    annotation_path: Path,
    dataset_id: str = "wider-face-val",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse the official WIDER FACE text annotations into image and box tables."""

    lines = annotation_path.read_text(encoding="utf-8").splitlines()
    image_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        relative_path = unicodedata.normalize(
            "NFC", Path(lines[index].strip()).as_posix()
        ).lstrip("./")
        relative_path_key = normalize_relative_path(relative_path)
        index += 1
        if not relative_path:
            continue
        if index >= len(lines):
            raise ValueError(f"Missing face count for WIDER image: {relative_path}")
        try:
            face_count = int(lines[index].strip())
        except ValueError as exc:
            raise ValueError(f"Invalid WIDER face count for {relative_path}") from exc
        index += 1
        image_id = hashlib.sha256(
            f"{dataset_id}:{relative_path_key}".encode("utf-8")
        ).hexdigest()[:16]
        image_rows.append(
            {
                "dataset_id": dataset_id,
                "imagem_id": image_id,
                "caminho_relativo": relative_path,
                "modelo_ia": "WIDER FACE",
                "tipo_prompt": pd.NA,
            }
        )
        valid_index = 0
        for _ in range(face_count):
            if index >= len(lines):
                raise ValueError(f"Unexpected end of WIDER annotations: {relative_path}")
            fields = lines[index].split()
            index += 1
            if len(fields) < 10:
                raise ValueError(f"Invalid WIDER face row for {relative_path}")
            x, y, w, h = map(float, fields[:4])
            invalid = int(fields[7])
            if invalid or w <= 0 or h <= 0:
                continue
            valid_index += 1
            truth_rows.append(
                {
                    "imagem_id": image_id,
                    "gt_face_id": f"{image_id}_gt{valid_index:03d}",
                    "bbox_x": max(0.0, x),
                    "bbox_y": max(0.0, y),
                    "bbox_w": w,
                    "bbox_h": h,
                }
            )
    return pd.DataFrame(image_rows), pd.DataFrame(truth_rows)


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
    predicted_by_image = {
        str(image_id): group.reset_index(drop=True)
        for image_id, group in faces.groupby("imagem_id", sort=False)
    }
    truth_by_image = {
        str(image_id): group.reset_index(drop=True)
        for image_id, group in truth.groupby("imagem_id", sort=False)
    }
    rows = []
    for image_id in images["imagem_id"]:
        image_key = str(image_id)
        predicted = predicted_by_image.get(image_key, faces.iloc[0:0])
        target = truth_by_image.get(image_key, truth.iloc[0:0])
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


def average_precision_at_iou(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    threshold: float = 0.5,
    confidence_col: str = "detector_confidence",
) -> float:
    """Calculate global interpolated AP for confidence-ranked detections."""

    required_predictions = {"imagem_id", confidence_col, *BOX_COLUMNS}
    required_truth = {"imagem_id", *BOX_COLUMNS}
    if missing := required_predictions - set(predictions):
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    if missing := required_truth - set(truth):
        raise ValueError(f"Truth is missing columns: {sorted(missing)}")
    if truth.empty:
        raise ValueError("AP50 requires at least one Ground Truth box")
    tie_breakers = ["imagem_id", *BOX_COLUMNS]
    ranked = predictions.sort_values(
        [confidence_col, *tie_breakers],
        ascending=[False, *([True] * len(tie_breakers))],
        kind="mergesort",
    )
    truth_by_image = {
        str(image_id): group[BOX_COLUMNS].to_numpy(dtype=float).tolist()
        for image_id, group in truth.groupby("imagem_id", sort=False)
    }
    matched: dict[str, set[int]] = {}
    tp_flags, fp_flags = [], []
    for prediction in ranked.itertuples(index=False):
        image_id = str(prediction.imagem_id)
        target = truth_by_image.get(image_id, [])
        used = matched.setdefault(image_id, set())
        best_index, best_iou = None, 0.0
        prediction_box = [getattr(prediction, column) for column in BOX_COLUMNS]
        for target_index, target_box in enumerate(target):
            if target_index in used:
                continue
            overlap = bbox_iou(prediction_box, target_box)
            if overlap > best_iou:
                best_index, best_iou = target_index, overlap
        if best_index is not None and best_iou >= threshold:
            used.add(best_index)
            tp_flags.append(1.0)
            fp_flags.append(0.0)
        else:
            tp_flags.append(0.0)
            fp_flags.append(1.0)
    if not tp_flags:
        return 0.0
    cumulative_tp = np.cumsum(tp_flags)
    cumulative_fp = np.cumsum(fp_flags)
    recall = cumulative_tp / len(truth)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1.0)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for position in range(len(precision) - 2, -1, -1):
        precision[position] = max(precision[position], precision[position + 1])
    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def compare_detectors(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    images: pd.DataFrame,
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe detector disagreement without treating either output as truth."""

    required = {"imagem_id", *BOX_COLUMNS}
    for label, table in (("primary", primary), ("secondary", secondary)):
        if missing := required - set(table):
            raise ValueError(f"{label} detections are missing columns: {sorted(missing)}")
    rows = []
    for image_id in images["imagem_id"].astype(str):
        first = primary[primary["imagem_id"].astype(str) == image_id].reset_index(drop=True)
        second = secondary[secondary["imagem_id"].astype(str) == image_id].reset_index(drop=True)
        matches = match_boxes(first, second, threshold=threshold)
        denominator = max(len(first) + len(second), 1)
        rows.append(
            {
                "imagem_id": image_id,
                "primary_count": len(first),
                "secondary_count": len(second),
                "count_difference": len(first) - len(second),
                "matched_boxes": len(matches),
                "mean_matched_iou": (
                    float(np.mean([match[2] for match in matches])) if matches else np.nan
                ),
                "unmatched_primary": len(first) - len(matches),
                "unmatched_secondary": len(second) - len(matches),
                "unmatched_rate": (len(first) + len(second) - 2 * len(matches)) / denominator,
            }
        )
    per_image = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "n_images": len(per_image),
                "exact_count_agreement_rate": float(
                    (per_image["count_difference"] == 0).mean()
                ),
                "mean_absolute_count_difference": float(
                    per_image["count_difference"].abs().mean()
                ),
                "mean_matched_iou": per_image["mean_matched_iou"].mean(),
                "mean_unmatched_rate": per_image["unmatched_rate"].mean(),
            }
        ]
    )
    return per_image, summary


def bootstrap_detector_disagreement(
    per_image: pd.DataFrame,
    *,
    iterations: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap image-level detector-disagreement metrics."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for _ in range(iterations):
        sample = per_image.iloc[rng.integers(0, len(per_image), len(per_image))]
        rows.append(
            {
                "exact_count_agreement_rate": float(
                    (sample["count_difference"] == 0).mean()
                ),
                "mean_absolute_count_difference": float(
                    sample["count_difference"].abs().mean()
                ),
                "mean_matched_iou": float(sample["mean_matched_iou"].mean()),
                "mean_unmatched_rate": float(sample["unmatched_rate"].mean()),
            }
        )
    samples = pd.DataFrame(rows)
    return pd.DataFrame(
        {
            "metric": samples.columns,
            "ci_2_5": samples.quantile(0.025).to_numpy(),
            "ci_97_5": samples.quantile(0.975).to_numpy(),
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
