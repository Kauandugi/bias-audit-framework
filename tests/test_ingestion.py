from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from biasauditfw.config import DatasetConfig
from biasauditfw.ingestion import discover_dataset
from conftest import make_image


def config(root: Path, **kwargs: object) -> DatasetConfig:
    return DatasetConfig(
        input_root=root,
        output_dir=root / "outputs_v2",
        dataset_id="fixture",
        **kwargs,
    )


def test_recursive_discovery_supports_depth_unicode_spaces_and_uppercase(
    tmp_path: Path,
) -> None:
    make_image(tmp_path / "nível com espaço" / "outro" / "A.PNG")
    make_image(tmp_path / "flat.jpg", (10, 20, 30))
    make_image(tmp_path / "outputs_v2" / "ignored.png", (30, 20, 10))

    discovery = discover_dataset(config(tmp_path, expected_images=2))

    assert len(discovery.images) == 2
    assert discovery.images["modelo_ia"].isna().all()
    assert discovery.images["tipo_prompt"].isna().all()
    assert discovery.images["imagem_id"].is_unique
    assert set(discovery.report["status_ingestao"]) == {"valid"}


def test_manifest_supplies_metadata_without_folder_inference(tmp_path: Path) -> None:
    image = make_image(tmp_path / "arbitrary" / "picture.webp")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "relative_path": image.relative_to(tmp_path).as_posix(),
                "source_image_id": "source-1",
                "modelo_ia": "Model X",
                "tipo_prompt": "Inclusive Prompts",
                "prompt_id": "P1",
                "pair_id": "X_P1",
                "analysis_eligible": "true",
            }
        ]
    ).to_csv(manifest, index=False)

    row = discover_dataset(config(tmp_path, manifest_path=manifest)).images.iloc[0]

    assert row["source_image_id"] == "source-1"
    assert row["modelo_ia"] == "Model X"
    assert row["analysis_eligible"] == True


def test_partial_manifest_keeps_unlisted_images_with_missing_metadata(
    tmp_path: Path,
) -> None:
    first = make_image(tmp_path / "first.png")
    make_image(tmp_path / "second.png", (1, 2, 3))
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [{"relative_path": first.name, "analysis_eligible": False}]
    ).to_csv(manifest, index=False)

    images = discover_dataset(config(tmp_path, manifest_path=manifest)).images

    assert len(images) == 2
    assert images["analysis_eligible"].astype("boolean").fillna(False).sum() == 0


def test_duplicate_manifest_and_missing_reference_are_rejected(tmp_path: Path) -> None:
    make_image(tmp_path / "image.png")
    duplicate_manifest = tmp_path / "duplicate.csv"
    pd.DataFrame(
        [{"relative_path": "image.png"}, {"relative_path": "./IMAGE.PNG"}]
    ).to_csv(duplicate_manifest, index=False)
    with pytest.raises(ValueError, match="duplicate relative_path"):
        discover_dataset(config(tmp_path, manifest_path=duplicate_manifest))

    missing_manifest = tmp_path / "missing.csv"
    pd.DataFrame([{"relative_path": "not-there.png"}]).to_csv(
        missing_manifest, index=False
    )
    with pytest.raises(ValueError, match="missing images"):
        discover_dataset(config(tmp_path, manifest_path=missing_manifest))


def test_duplicate_content_can_fail_or_keep_first(tmp_path: Path) -> None:
    original = make_image(tmp_path / "a.png")
    (tmp_path / "nested").mkdir()
    duplicate = tmp_path / "nested" / "b.png"
    duplicate.write_bytes(original.read_bytes())

    with pytest.raises(ValueError, match="Duplicate image content"):
        discover_dataset(config(tmp_path))

    discovery = discover_dataset(config(tmp_path, duplicate_policy="first"))
    assert len(discovery.images) == 1
    assert "duplicate" in set(discovery.report["status_ingestao"])


def test_corrupt_empty_missing_and_expected_count_errors(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No supported images"):
        discover_dataset(config(empty))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        discover_dataset(config(tmp_path / "absent"))

    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / "bad.png").write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid image"):
        discover_dataset(config(corrupt))

    valid = tmp_path / "valid"
    make_image(valid / "one.png")
    with pytest.raises(ValueError, match="Expected 2"):
        discover_dataset(config(valid, expected_images=2))


def test_symlink_is_not_processed_when_supported(tmp_path: Path) -> None:
    target = make_image(tmp_path / "target.png")
    link = tmp_path / "linked.png"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are not available in this environment")

    images = discover_dataset(config(tmp_path)).images
    assert images["caminho_relativo"].tolist() == ["target.png"]
