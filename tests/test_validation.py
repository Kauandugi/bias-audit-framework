from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biasauditfw.validation import (
    bbox_iou,
    bootstrap_detector,
    coco_to_dataframe,
    concordance_report,
    evaluate_detector,
    harmonize_gender,
    harmonize_race,
    match_boxes,
)


BOXES = ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]


def test_iou_and_one_to_one_matching() -> None:
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert bbox_iou([0, 0, 2, 2], [5, 5, 2, 2]) == 0
    predicted = pd.DataFrame([[0, 0, 10, 10], [1, 1, 8, 8]], columns=BOXES)
    truth = pd.DataFrame([[0, 0, 10, 10]], columns=BOXES)
    matches = match_boxes(predicted, truth, threshold=0.5)
    assert len(matches) == 1
    assert match_boxes(pd.DataFrame(columns=BOXES), truth) == []


def test_coco_uses_relative_path_or_source_id_not_basename(tmp_path: Path) -> None:
    images = pd.DataFrame(
        {
            "imagem_id": ["a", "b"],
            "caminho_relativo": ["one/shared.png", "two/shared.png"],
            "source_image_id": ["SOURCE-A", "SOURCE-B"],
        }
    )
    coco = {
        "images": [
            {"id": 1, "file_name": "two/shared.png"},
            {
                "id": 2,
                "file_name": "unrelated/name.png",
                "source_image_id": "SOURCE-A",
            },
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "bbox": [1, 2, 3, 4]},
            {"id": 2, "image_id": 2, "bbox": [5, 6, 7, 8]},
        ],
    }
    path = tmp_path / "truth.json"
    path.write_text(json.dumps(coco), encoding="utf-8")

    truth = coco_to_dataframe(path, images, "R1")

    assert truth["imagem_id"].tolist() == ["b", "a"]
    assert truth["gt_face_id"].is_unique

    coco["images"][0]["file_name"] = "unknown/shared.png"
    path.write_text(json.dumps(coco), encoding="utf-8")
    with pytest.raises(ValueError, match="absent from inventory"):
        coco_to_dataframe(path, images, "R1")


def test_detector_metrics_and_bootstrap() -> None:
    images = pd.DataFrame(
        {
            "imagem_id": ["a", "b"],
            "modelo_ia": ["M1", "M2"],
            "tipo_prompt": ["Neutral", "Inclusive"],
        }
    )
    faces = pd.DataFrame(
        [
            {"imagem_id": "a", **dict(zip(BOXES, [0, 0, 10, 10]))},
            {"imagem_id": "b", **dict(zip(BOXES, [20, 20, 5, 5]))},
        ]
    )
    truth = pd.DataFrame(
        [
            {"imagem_id": "a", **dict(zip(BOXES, [0, 0, 10, 10]))},
            {"imagem_id": "b", **dict(zip(BOXES, [0, 0, 5, 5]))},
        ]
    )
    per_image, report = evaluate_detector(faces, truth, images)
    confidence = bootstrap_detector(per_image, iterations=30, seed=1)

    overall = report.loc[report["grupo"] == "overall"].iloc[0]
    assert overall["tp"] == 1
    assert overall["fp"] == 1
    assert overall["fn"] == 1
    assert overall["f1"] == pytest.approx(0.5)
    assert set(confidence["metric"]) == {"precision", "recall", "f1"}


def test_concordance_handles_harmonization_and_unmapped_values() -> None:
    merged = pd.DataFrame(
        {
            "deepface_raca": ["asian", "white", "unknown"],
            "fairface_raca": ["East Asian", "White", "Other"],
            "deepface_genero": ["Man", "Woman", "unknown"],
            "fairface_genero": ["Male", "Female", "Other"],
        }
    )
    report, matrices = concordance_report(merged)

    assert harmonize_race("Southeast_Asian") == "asian"
    assert harmonize_gender("Woman") == "female"
    assert (report["agreement"] == 1).all()
    assert matrices["race"].to_numpy().sum() == 2

    empty, matrices = concordance_report(
        pd.DataFrame(
            {
                "deepface_raca": ["unknown"],
                "fairface_raca": ["other"],
                "deepface_genero": ["unknown"],
                "fairface_genero": ["other"],
            }
        )
    )
    assert (empty["n"] == 0).all()
    assert matrices["gender"].empty

