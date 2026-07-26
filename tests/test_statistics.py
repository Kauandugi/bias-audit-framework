from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biasauditfw.statistics import (
    face_diversity_by_image,
    human_clip_correlations,
    human_interrater_report,
    inference_eligibility,
    mann_whitney_reports,
    normalized_simpson,
    per_model_mann_whitney,
    rank_biserial_from_u,
    require_unique_images,
    wilcoxon_reports,
)


def analysis_fixture() -> pd.DataFrame:
    rows = []
    for pair in range(1, 5):
        for group, shift in (("Neutral Prompts", 0.0), ("Inclusive Prompts", 0.3)):
            rows.append(
                {
                    "imagem_id": f"{pair}-{group[0]}",
                    "content_sha256": f"{pair}-{group}",
                    "source_image_id": f"S{pair}-{group[0]}",
                    "pair_id": f"P{pair}",
                    "tipo_prompt": group,
                    "modelo_ia": "A" if pair < 3 else "B",
                    "analysis_eligible": True,
                    "clip_racial_diversity_margin": pair * 0.1 + shift,
                    "clip_gender_diversity_margin": pair * 0.05 + shift,
                }
            )
    return pd.DataFrame(rows)


def test_primary_inference_and_holm_reports() -> None:
    images = analysis_fixture()
    eligible, reason = inference_eligibility(images)
    mann = mann_whitney_reports(images)
    paired = wilcoxon_reports(images)

    assert eligible and reason == "eligible"
    assert set(mann["metric"]) == {
        "clip_racial_diversity_margin",
        "clip_gender_diversity_margin",
    }
    assert mann["p_value_holm"].between(0, 1).all()
    assert (paired["n_pairs"] == 4).all()
    assert np.allclose(paired["median_paired_difference"], 0.3)
    assert rank_biserial_from_u(0, 4, 4) == -1


def test_inference_rejects_pseudoreplication_and_invalid_pairing() -> None:
    images = analysis_fixture()
    duplicated = pd.concat([images, images.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique imagem_id"):
        mann_whitney_reports(duplicated)

    duplicated_content = images.copy()
    duplicated_content.loc[1, "content_sha256"] = duplicated_content.loc[
        0, "content_sha256"
    ]
    with pytest.raises(ValueError, match="pseudoreplication"):
        require_unique_images(duplicated_content)

    invalid = images.copy()
    invalid.loc[0, "pair_id"] = pd.NA
    eligible, reason = inference_eligibility(invalid)
    assert not eligible
    assert "pair_id" in reason

    no_manifest = images.copy()
    no_manifest["analysis_eligible"] = pd.NA
    eligible, reason = inference_eligibility(no_manifest)
    assert not eligible
    assert "No images" in reason


def test_per_model_family_and_zero_difference_wilcoxon() -> None:
    images = analysis_fixture()
    images["clip_racial_diversity_margin"] = np.tile([0.1, 0.1], 4)
    images["clip_gender_diversity_margin"] = np.tile([0.2, 0.2], 4)
    paired = wilcoxon_reports(images)
    model_report = per_model_mann_whitney(images)

    assert (paired["p_value"] == 1).all()
    assert len(model_report) == 4
    assert model_report["p_value_holm_family"].between(0, 1).all()


def test_simpson_indices_have_known_results_and_require_two_faces() -> None:
    assert np.isnan(normalized_simpson(pd.Series(["white"]), 6))
    assert normalized_simpson(pd.Series(["a", "b"]), 2) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="at least two"):
        normalized_simpson(pd.Series(["a", "b"]), 1)

    faces = pd.DataFrame(
        {
            "imagem_id": ["one", "two", "two"],
            "raca": ["white", "white", "black"],
            "genero": ["Man", "Man", "Woman"],
        }
    )
    report = face_diversity_by_image(faces).set_index("imagem_id")
    assert np.isnan(report.loc["one", "deepface_racial_simpson"])
    assert report.loc["two", "deepface_gender_simpson"] == pytest.approx(1.0)


def test_human_kappa_and_clip_correlations_match_dimensions() -> None:
    images = analysis_fixture().iloc[:4].copy()
    human = pd.DataFrame(
        {
            "Image_id": images["source_image_id"],
            "R1_Racial_Diversity": [1, 2, 3, 4],
            "R2_Racial_Diversity": [1, 2, 3, 4],
            "R1_Gender_Representation": [4, 3, 2, 1],
            "R2_Gender_Representation": [4, 3, 2, 1],
        }
    )
    interrater = human_interrater_report(human)
    correlations = human_clip_correlations(human, images)

    assert (interrater["weighted_kappa"] == 1).all()
    assert len(correlations) == 2
    assert correlations["p_value_holm"].between(0, 1).all()
