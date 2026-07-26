from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from biasauditfw.reporting import descriptive_report, export_statistical_reports
from test_statistics import analysis_fixture


def test_reports_export_inference_when_manifest_metadata_is_valid(
    tmp_path: Path,
) -> None:
    images = analysis_fixture()
    faces = pd.DataFrame(
        {
            "imagem_id": [images.iloc[0]["imagem_id"], images.iloc[0]["imagem_id"]],
            "raca": ["white", "black"],
            "genero": ["Man", "Woman"],
        }
    )
    status = export_statistical_reports(images, faces, tmp_path)

    assert status == {"eligible": True, "reason": "eligible"}
    assert (tmp_path / "mann_whitney_primary.csv").exists()
    assert (tmp_path / "wilcoxon_sensitivity.csv").exists()
    assert len(descriptive_report(images)) > 2


def test_reports_skip_inference_without_manifest_metadata(tmp_path: Path) -> None:
    images = analysis_fixture()
    images["analysis_eligible"] = pd.NA
    faces = pd.DataFrame(columns=["imagem_id", "raca", "genero"])
    (tmp_path / "mann_whitney_primary.csv").write_text("stale", encoding="utf-8")
    status = export_statistical_reports(images, faces, tmp_path)

    saved = json.loads(
        (tmp_path / "inference_status.json").read_text(encoding="utf-8")
    )
    assert not status["eligible"]
    assert saved == status
    assert (tmp_path / "descriptive_diversity.csv").exists()
    assert not (tmp_path / "mann_whitney_primary.csv").exists()
