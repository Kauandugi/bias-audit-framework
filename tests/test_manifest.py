from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from biasauditfw.manifest import build_path_manifest, build_sidecar_manifest


def _case(root: Path, model: str, prompt_id: str, prompt_type: str) -> None:
    directory = root / model / prompt_type / prompt_id
    directory.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(len(prompt_id), 10, 20)).save(
        directory / "image.png"
    )
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "prompt_id": prompt_id,
                "prompt_type": prompt_type,
                "model": model,
            }
        ),
        encoding="utf-8",
    )


def test_sidecar_manifest_builds_complete_model_scoped_pairs(tmp_path: Path) -> None:
    for model in ("Model A", "Model B"):
        _case(tmp_path, model, "P1", "neutral")
        _case(tmp_path, model, "P1", "inclusive")

    manifest = build_sidecar_manifest(
        tmp_path, dataset_id="replication", expected_images=4
    )

    assert len(manifest) == 4
    assert manifest["pair_id"].nunique() == 2
    assert manifest["source_image_id"].is_unique
    assert manifest["analysis_eligible"].all()
    assert set(manifest["tipo_prompt"]) == {
        "Neutral Prompts",
        "Inclusive Prompts",
    }


def test_sidecar_manifest_rejects_incomplete_pair(tmp_path: Path) -> None:
    _case(tmp_path, "Model A", "P1", "neutral")

    with pytest.raises(ValueError, match="neutral and one inclusive"):
        build_sidecar_manifest(tmp_path, dataset_id="replication")


def test_sidecar_manifest_rejects_missing_metadata(tmp_path: Path) -> None:
    Image.new("RGB", (8, 8)).save(tmp_path / "image.png")

    with pytest.raises(FileNotFoundError, match="Missing sidecar metadata"):
        build_sidecar_manifest(tmp_path, dataset_id="replication")


def test_path_manifest_keeps_incomplete_pair_but_excludes_it_from_inference(
    tmp_path: Path,
) -> None:
    for group in ("Neutro", "Inclusivo"):
        directory = tmp_path / "Model A" / group / "prompt 1"
        directory.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(directory / f"{group}.png")
    incomplete = tmp_path / "Model A" / "Neutro" / "prompt 2"
    incomplete.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(incomplete / "neutral.png")

    manifest = build_path_manifest(tmp_path, dataset_id="replication")

    assert len(manifest) == 3
    assert manifest["analysis_eligible"].sum() == 2
    assert not manifest.loc[manifest["prompt_id"] == "P2", "analysis_eligible"].any()


def test_path_manifest_accepts_content_valid_image_with_unrecognized_suffix(
    tmp_path: Path,
) -> None:
    for group in ("Neutro", "Inclusivo"):
        directory = tmp_path / "Model A" / group / "prompt 1"
        directory.mkdir(parents=True)
        path = directory / f"{group}._"
        Image.new("RGB", (8, 8)).save(path, format="PNG")

    manifest = build_path_manifest(
        tmp_path,
        dataset_id="replication",
        expected_images=2,
    )

    assert len(manifest) == 2
    assert manifest["analysis_eligible"].all()
