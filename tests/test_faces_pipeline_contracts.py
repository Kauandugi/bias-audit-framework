from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pytest

from biasauditfw.clip_diversity import CLIP_OUTPUT_COLUMNS
from biasauditfw.contracts import FACE_COLUMNS, IMAGE_COLUMNS, validate_contract
from biasauditfw.faces import analyze_faces, normalize_deepface_results
from biasauditfw.pipeline import (
    export_results,
    process_dataset,
    select_smoke_inventory,
)
from conftest import make_image


def clip_scores(offset: float = 0.0) -> dict[str, float]:
    return {
        "clip_racial_diversity_cosine": 0.4 + offset,
        "clip_racial_homogeneity_cosine": 0.1,
        "clip_racial_diversity_margin": 0.3 + offset,
        "clip_gender_diversity_cosine": 0.2,
        "clip_gender_homogeneity_cosine": 0.3,
        "clip_gender_diversity_margin": -0.1,
    }


def inventory(tmp_path: Path, count: int = 2) -> pd.DataFrame:
    rows = []
    for index in range(count):
        path = make_image(tmp_path / f"image-{index}.png", (index, 20, 30))
        rows.append(
            {
                "dataset_id": "fixture",
                "imagem_id": f"img{index}",
                "content_sha256": f"{index:064x}",
                "caminho_relativo": path.name,
                "nome_arquivo": path.name,
                "source_image_id": f"S{index}",
                "prompt_id": "P1",
                "pair_id": "PAIR1",
                "tipo_prompt": "Inclusive" if index else "Neutral",
                "modelo_ia": "Model",
                "analysis_eligible": True,
                "absolute_path": path,
            }
        )
    return pd.DataFrame(rows)


def test_normalize_deepface_zero_one_many_and_defensive_boxes() -> None:
    assert normalize_deepface_results(None) == []
    raw = [
        {
            "region": {"x": 20, "y": 8, "w": 5, "h": 6},
            "dominant_race": "white",
            "race": {"white": 70},
            "dominant_gender": "Man",
            "gender": {"Man": 80},
        },
        {
            "region": {"x": -2, "y": 1, "w": 7, "h": 8},
            "dominant_race": "black",
            "race": {"black": 90},
            "dominant_gender": "Woman",
            "gender": {"Woman": 60},
        },
        {"region": {"x": 1, "y": 2, "w": 0, "h": 3}},
        "ignored",
    ]
    faces = normalize_deepface_results(raw)
    assert len(faces) == 2
    assert faces[0]["bbox_x"] == 0
    assert faces[0]["confianca_raca"] == 90.0
    assert faces[1]["genero"] == "Man"


def test_analyze_faces_distinguishes_no_face_from_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class NoFace:
        @staticmethod
        def analyze(**_: object):
            raise ValueError("Face could not be detected")

    monkeypatch.setitem(sys.modules, "deepface", SimpleNamespace(DeepFace=NoFace))
    assert analyze_faces(tmp_path / "none.png") == ([], "no_face", None)

    class Broken:
        @staticmethod
        def analyze(**_: object):
            raise RuntimeError("backend unavailable")

    monkeypatch.setitem(sys.modules, "deepface", SimpleNamespace(DeepFace=Broken))
    faces, status, error = analyze_faces(tmp_path / "broken.png")
    assert faces == []
    assert status == "error"
    assert "backend unavailable" in str(error)


def test_pipeline_preserves_1_to_n_and_clip_only_at_image_level(
    tmp_path: Path,
) -> None:
    def face_analyzer(path: Path):
        if path.name.endswith("0.png"):
            return [], "no_face", None
        faces = [
            {
                "bbox_x": 1,
                "bbox_y": 2,
                "bbox_w": 5,
                "bbox_h": 6,
                "raca": "black",
                "confianca_raca": 80.0,
                "genero": "Woman",
                "confianca_genero": 70.0,
            },
            {
                "bbox_x": 8,
                "bbox_y": 2,
                "bbox_w": 4,
                "bbox_h": 5,
                "raca": "white",
                "confianca_raca": 75.0,
                "genero": "Man",
                "confianca_genero": 65.0,
            },
        ]
        return faces, "detected", None

    images, faces = process_dataset(inventory(tmp_path), face_analyzer, lambda _: clip_scores())

    assert list(images.columns) == IMAGE_COLUMNS
    assert list(faces.columns) == FACE_COLUMNS
    assert images["num_faces"].tolist() == [0, 2]
    assert len(faces) == 2
    assert not any(column.startswith("clip_") for column in faces)
    validate_contract(images, faces, expected_images=2)


def test_smoke_selection_uses_balanced_pairs_across_models() -> None:
    rows = []
    for model in ("Model A", "Model B", "Model C"):
        for prompt_id in ("P1", "P2"):
            pair_id = f"{model}-{prompt_id}"
            for prompt_type in ("Neutral Prompts", "Inclusive Prompts"):
                rows.append(
                    {
                        "analysis_eligible": True,
                        "modelo_ia": model,
                        "pair_id": pair_id,
                        "tipo_prompt": prompt_type,
                        "caminho_relativo": (
                            f"{model}/{prompt_type}/{prompt_id}.png"
                        ),
                    }
                )
    selected = select_smoke_inventory(pd.DataFrame(rows), max_images=4)

    assert len(selected) == 4
    assert selected["modelo_ia"].nunique() == 2
    assert set(selected["tipo_prompt"]) == {
        "Neutral Prompts",
        "Inclusive Prompts",
    }
    assert selected.groupby("pair_id").size().eq(2).all()


def test_pipeline_records_clip_errors_and_rejects_missing_scores(tmp_path: Path) -> None:
    images, faces = process_dataset(
        inventory(tmp_path, 1),
        lambda _: ([], "no_face", None),
        lambda _: {},
    )
    assert images.iloc[0]["status_clip"] == "error"
    assert images[list(CLIP_OUTPUT_COLUMNS)].isna().all().all()
    validate_contract(images, faces)


def test_contract_rejects_bad_counts_ranges_and_legacy_columns(tmp_path: Path) -> None:
    images, faces = process_dataset(
        inventory(tmp_path),
        lambda _: ([], "no_face", None),
        lambda _: clip_scores(),
    )
    bad = images.copy()
    bad.loc[0, "clip_racial_diversity_cosine"] = 1.5
    with pytest.raises(ValueError, match="outside"):
        validate_contract(bad, faces)

    bad = images.copy()
    bad["clip_latin_american"] = 0.2
    with pytest.raises(ValueError, match="schema"):
        validate_contract(bad, faces)


def test_export_writes_schema_metadata_and_canonical_files(tmp_path: Path) -> None:
    images, faces = process_dataset(
        inventory(tmp_path),
        lambda _: ([], "no_face", None),
        lambda _: clip_scores(),
    )
    output = tmp_path / "export"
    report = pd.DataFrame([{"status_ingestao": "valid"}])
    export_results(images, faces, report, output, {"device": "cpu"}, 2)

    assert (output / "audit_images.csv").exists()
    assert (output / "audit_faces.csv").exists()
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "2.0"
    assert metadata["device"] == "cpu"
