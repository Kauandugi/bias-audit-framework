from __future__ import annotations

import pandas as pd
import pytest

from biasauditfw.human_forms import (
    FORM_SCORE_COLUMNS,
    filter_consented_responses,
    google_forms_responses_to_long,
    validate_form_image_map,
)
from biasauditfw.statistics import aggregate_human_ratings, ordinal_krippendorff_report


LABELS = {
    "racial_score": "Diversidade racial",
    "gender_score": "Equilíbrio de gênero",
    "cultural_score": "Contexto latino-americano",
    "body_score": "Diversidade corporal",
    "quality_score": "Qualidade e coerência visual",
}


def image_map() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "form_position": [1, 2],
            "imagem_id": ["img-a", "img-b"],
            "dataset_id": ["original-64", "vitor-64"],
        }
    )


def responses() -> pd.DataFrame:
    rows = []
    for participant, offset in (("first", 0), ("second", 1)):
        row: dict[str, object] = {
            "Carimbo de data/hora": participant,
            "Li as informações acima e concordo em participar.": "Sim, concordo",
            "Qual é a sua faixa etária?": "18 a 24 anos",
        }
        for position in (1, 2):
            for score_column in FORM_SCORE_COLUMNS.values():
                row[f"Imagem {position} [{LABELS[score_column]}]"] = min(
                    5, position + 2 + offset
                )
        rows.append(row)
    return pd.DataFrame(rows)


def test_google_forms_export_becomes_private_long_table() -> None:
    long = google_forms_responses_to_long(
        responses(), image_map(), expected_images=2
    )

    assert len(long) == 4
    assert set(long["participant_id"]) == {"P001", "P002"}
    assert "Carimbo de data/hora" not in long
    assert "Qual é a sua faixa etária?" not in long
    assert set(FORM_SCORE_COLUMNS.values()).issubset(long.columns)
    assert long.groupby("participant_id")["imagem_id"].nunique().eq(2).all()

    aggregate = aggregate_human_ratings(long)
    agreement = ordinal_krippendorff_report(long)
    assert {
        "human_cultural_median",
        "human_body_median",
        "human_quality_median",
    }.issubset(aggregate.columns)
    assert set(agreement["dimension"]) == set(FORM_SCORE_COLUMNS)


def test_google_forms_import_excludes_responses_without_consent() -> None:
    exported = responses()
    refused = exported.iloc[[0]].copy().astype(object)
    refused.loc[:, "Li as informações acima e concordo em participar."] = (
        "Não concordo"
    )
    for column in refused.columns:
        if str(column).startswith("Imagem "):
            refused.loc[:, column] = pd.NA
    exported = pd.concat([exported, refused], ignore_index=True)

    filtered, summary = filter_consented_responses(exported)
    long = google_forms_responses_to_long(exported, image_map(), expected_images=2)

    assert len(filtered) == 2
    assert len(long) == 4
    assert summary["responses_total"] == 3
    assert summary["responses_consented"] == 2
    assert summary["responses_declined"] == 1
    assert summary["responses_excluded"] == 1


def test_google_forms_import_rejects_missing_or_invalid_scores() -> None:
    missing = responses().drop(columns=["Imagem 2 [Diversidade corporal]"])
    with pytest.raises(ValueError, match="Missing Google Forms score columns"):
        google_forms_responses_to_long(missing, image_map(), expected_images=2)

    invalid = responses()
    invalid.loc[0, "Imagem 1 [Diversidade racial]"] = 6
    with pytest.raises(ValueError, match="Invalid score"):
        google_forms_responses_to_long(invalid, image_map(), expected_images=2)


def test_form_map_requires_consecutive_unique_positions_and_images() -> None:
    duplicate = image_map()
    duplicate.loc[1, "imagem_id"] = "img-a"
    with pytest.raises(ValueError, match="imagem_id must be unique"):
        validate_form_image_map(duplicate, expected_images=2)

    gap = image_map()
    gap.loc[1, "form_position"] = 3
    with pytest.raises(ValueError, match="consecutive"):
        validate_form_image_map(gap, expected_images=2)
