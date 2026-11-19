from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from biasauditfw.neural import run_fairface_on_detected_faces
from biasauditfw.validation import (
    average_precision_at_iou,
    compare_detectors,
    wider_face_to_dataframes,
)

from conftest import make_image
from scripts.run_detector_validation import bootstrap_detector_disagreement


def test_wider_parser_ignores_invalid_boxes(tmp_path: Path) -> None:
    annotations = tmp_path / "wider.txt"
    annotations.write_text(
        "0--Parade/example.jpg\n"
        "2\n"
        "10 20 30 40 0 0 0 0 0 0\n"
        "1 2 3 4 0 0 0 1 0 0\n"
        "1--Handshaking/empty.jpg\n"
        "0\n",
        encoding="utf-8",
    )
    images, truth = wider_face_to_dataframes(annotations)
    assert len(images) == 2
    assert len(truth) == 1
    assert truth.iloc[0]["bbox_w"] == 30
    assert images.iloc[0]["caminho_relativo"] == "0--Parade/example.jpg"


def test_ap50_and_detector_disagreement_known_cases() -> None:
    images = pd.DataFrame([{"imagem_id": "a"}, {"imagem_id": "b"}])
    truth = pd.DataFrame(
        [{"imagem_id": "a", "bbox_x": 0, "bbox_y": 0, "bbox_w": 10, "bbox_h": 10}]
    )
    primary = pd.DataFrame(
        [
            {
                "imagem_id": "a",
                "bbox_x": 0,
                "bbox_y": 0,
                "bbox_w": 10,
                "bbox_h": 10,
                "detector_confidence": 0.9,
            },
            {
                "imagem_id": "b",
                "bbox_x": 20,
                "bbox_y": 20,
                "bbox_w": 5,
                "bbox_h": 5,
                "detector_confidence": 0.1,
            },
        ]
    )
    assert average_precision_at_iou(primary, truth) == pytest.approx(1.0)
    secondary = pd.DataFrame(
        [{"imagem_id": "a", "bbox_x": 0, "bbox_y": 0, "bbox_w": 10, "bbox_h": 10}]
    )
    per_image, summary = compare_detectors(primary, secondary, images)
    assert per_image.set_index("imagem_id").loc["a", "matched_boxes"] == 1
    assert summary.iloc[0]["exact_count_agreement_rate"] == pytest.approx(0.5)


def test_ap50_is_invariant_to_input_order_when_confidences_are_tied() -> None:
    truth = pd.DataFrame(
        [
            {"imagem_id": "a", "bbox_x": 0, "bbox_y": 0, "bbox_w": 10, "bbox_h": 10},
            {"imagem_id": "b", "bbox_x": 0, "bbox_y": 0, "bbox_w": 10, "bbox_h": 10},
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "imagem_id": "b", "bbox_x": 20, "bbox_y": 20,
                "bbox_w": 5, "bbox_h": 5, "detector_confidence": 0.9,
            },
            {
                "imagem_id": "a", "bbox_x": 0, "bbox_y": 0,
                "bbox_w": 10, "bbox_h": 10, "detector_confidence": 0.9,
            },
            {
                "imagem_id": "b", "bbox_x": 0, "bbox_y": 0,
                "bbox_w": 10, "bbox_h": 10, "detector_confidence": 0.9,
            },
        ]
    )

    expected = average_precision_at_iou(predictions, truth)
    shuffled = predictions.sample(frac=1, random_state=42).reset_index(drop=True)

    assert average_precision_at_iou(shuffled, truth) == pytest.approx(expected)


def test_fairface_uses_detected_face_ids_and_crops(tmp_path: Path) -> None:
    image_path = make_image(tmp_path / "scene.png")
    images = pd.DataFrame(
        [{"imagem_id": "img", "caminho_relativo": image_path.name}]
    )
    faces = pd.DataFrame(
        [
            {
                "imagem_id": "img",
                "face_id": "img_f001",
                "bbox_x": 2,
                "bbox_y": 3,
                "bbox_w": 8,
                "bbox_h": 7,
                "raca": "white",
                "genero": "Woman",
            }
        ]
    )

    def oracle(crop):
        assert crop.size == (8, 7)
        return {
            "fairface_raca": "White",
            "fairface_conf_raca": 0.8,
            "fairface_genero": "Female",
            "fairface_conf_genero": 0.9,
        }

    result = run_fairface_on_detected_faces(oracle, faces, images, tmp_path)
    assert result.iloc[0]["face_id"] == "img_f001"
    assert result.iloc[0]["deepface_raca"] == "white"


def test_bootstrap_detector_disagreement_returns_ordered_intervals() -> None:
    per_image = pd.DataFrame(
        {
            "count_difference": [0, 1, -1],
            "mean_matched_iou": [0.9, 0.8, 0.7],
            "unmatched_rate": [0.0, 0.2, 0.4],
        }
    )

    result = bootstrap_detector_disagreement(per_image, iterations=100, seed=7)

    assert set(result["metric"]) == {
        "exact_count_agreement_rate",
        "mean_absolute_count_difference",
        "mean_matched_iou",
        "mean_unmatched_rate",
    }
    assert (result["ci_2_5"] <= result["ci_97_5"]).all()
