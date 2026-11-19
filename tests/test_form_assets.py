from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from scripts.export_unesp_form_assets import export_form_assets


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def test_export_form_assets_blinds_filenames(tmp_path: Path) -> None:
    original = tmp_path / "original"
    replication = tmp_path / "replication"
    _write_image(original / "secret" / "neutral.png", (1, 2, 3))
    _write_image(replication / "hidden" / "inclusive.jpg", (4, 5, 6))
    panel = pd.DataFrame(
        [
            {
                "form_code": "A1",
                "form_position": 1,
                "dataset_id": "original-64",
                "caminho_relativo": "secret/neutral.png",
            },
            {
                "form_code": "A1",
                "form_position": 2,
                "dataset_id": "vitor-64",
                "caminho_relativo": "hidden/inclusive.jpg",
            },
        ]
    )
    panel_csv = tmp_path / "panel.csv"
    panel.to_csv(panel_csv, index=False)

    result = export_form_assets(
        [panel_csv],
        {"original-64": original.resolve(), "vitor-64": replication.resolve()},
        tmp_path / "assets",
    )

    assert result["asset_path"].tolist() == ["A1/imagem_01.png", "A1/imagem_02.jpg"]
    assert not ({"dataset_id", "caminho_relativo"} & set(result))
    assert (tmp_path / "assets" / "A1" / "imagem_01.png").is_file()


def test_export_form_assets_preflights_before_copying(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write_image(root / "present.png", (1, 2, 3))
    panel = pd.DataFrame(
        [
            {
                "form_code": "A1",
                "form_position": 1,
                "dataset_id": "dataset",
                "caminho_relativo": "present.png",
            },
            {
                "form_code": "A1",
                "form_position": 2,
                "dataset_id": "dataset",
                "caminho_relativo": "missing.png",
            },
        ]
    )
    panel_csv = tmp_path / "panel.csv"
    panel.to_csv(panel_csv, index=False)
    output = tmp_path / "assets"

    with pytest.raises(FileNotFoundError):
        export_form_assets([panel_csv], {"dataset": root.resolve()}, output)

    assert not output.exists()
