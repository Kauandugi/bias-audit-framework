from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_unesp_human_ratings.py"
SPEC = importlib.util.spec_from_file_location("prepare_unesp_human_ratings", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _ratings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": ["P001", "P001", "P002", "P002"],
            "imagem_id": ["I1", "I2", "I1", "I2"],
        }
    )


def test_metadata_authorizes_only_aggregate_outputs() -> None:
    metadata = module.build_validation_metadata(
        _ratings(),
        {"responses_total": 3, "responses_consented": 2, "consent_column": "x"},
        form_code="UNESP-2026",
        authorized_for_aggregate_use=True,
        authorization_basis="Recorded institutional authorization",
        ethics_opinion_number="8.791.889",
        ethics_opinion_date="2026-10-08",
    )

    assert metadata["usable_as_final_evidence"] is True
    assert metadata["publication_scope"] == "aggregate_only"
    assert metadata["individual_data_published"] is False
    assert metadata["profile_fields_exported"] is False
    assert "consent_column" not in metadata


def test_metadata_requires_basis_for_authorized_use() -> None:
    with pytest.raises(ValueError, match="authorization-basis"):
        module.build_validation_metadata(
            _ratings(),
            {},
            form_code="UNESP-2026",
            authorized_for_aggregate_use=True,
            authorization_basis=None,
            ethics_opinion_number=None,
            ethics_opinion_date=None,
        )


def test_frozen_public_human_artifacts_are_aggregate_only() -> None:
    root = Path(__file__).parents[1] / "data" / "human-unesp-aggregate"
    aggregate = pd.read_csv(root / "human_ratings_aggregate.csv")
    agreement = pd.read_csv(root / "human_interrater_agreement.csv")
    effects = pd.read_csv(root / "human_paired_prompt_effects.csv")
    convergence = pd.read_csv(root / "human_framework_convergence.csv")
    metadata = json.loads(
        (root / "human_validation_metadata.json").read_text(encoding="utf-8")
    )

    forbidden = {
        "participant_id",
        "form_position",
        "timestamp",
        "email",
        "age",
        "institution",
        "consent",
        "consentimento",
        "country",
        "race",
        "participant_gender",
    }
    assert len(aggregate) == 48
    assert forbidden.isdisjoint({column.casefold() for column in aggregate})
    assert set(aggregate.filter(regex=r"_n$").stack().astype(int)) == {70}
    assert set(convergence["dataset_id"]) == {"original-64", "vitor-64"}
    assert metadata["responses_total"] == 72
    assert metadata["responses_consented"] == 70
    assert metadata["responses_declined"] == 2
    assert metadata["responses_excluded"] == 2
    assert metadata["participants"] == 70
    assert metadata["images"] == 48
    assert metadata["private_rows"] == 3360
    assert metadata["usable_as_final_evidence"] is True
    assert metadata["publication_scope"] == "aggregate_only"
    assert metadata["individual_data_published"] is False

    alpha = agreement.set_index("dimension")["krippendorff_alpha_ordinal"]
    assert alpha["racial"] == pytest.approx(0.3011753819)
    assert alpha["gender"] == pytest.approx(0.1657763817)
    assert alpha["cultural"] == pytest.approx(0.2536201264)
    assert alpha["body"] == pytest.approx(0.2881368083)
    assert alpha["quality"] == pytest.approx(0.1422071470)

    effect_index = effects.set_index(["human_corpus_id", "dimension"])
    assert effect_index.loc[
        ("original-64", "racial"), "p_value_holm"
    ] == pytest.approx(0.28125)
    assert effect_index.loc[
        ("original-64", "gender"), "p_value_holm"
    ] == pytest.approx(1.0)
    assert effect_index.loc[
        ("vitor-form-24", "racial"), "p_value_holm"
    ] == pytest.approx(0.40625)
    assert effect_index.loc[
        ("vitor-form-24", "gender"), "p_value_holm"
    ] == pytest.approx(0.0625)

    convergence_index = convergence.set_index(
        ["dataset_id", "dimension", "automated_metric"]
    )
    original_simpson = convergence_index.loc[
        ("original-64", "racial", "deepface_racial_simpson")
    ]
    replication_simpson = convergence_index.loc[
        ("vitor-64", "racial", "deepface_racial_simpson")
    ]
    assert original_simpson["spearman_p_value_holm"] == pytest.approx(
        0.1434161646
    )
    assert original_simpson[
        "partial_spearman_p_value_face_count_holm"
    ] == pytest.approx(0.0654331934)
    assert replication_simpson["spearman_p_value_holm"] == pytest.approx(
        0.1229110020
    )
    assert replication_simpson[
        "partial_spearman_p_value_face_count_holm"
    ] == pytest.approx(0.0337619026)

    for filename in (
        "human_ratings_aggregate.csv",
        "human_interrater_agreement.csv",
        "human_paired_prompt_effects.csv",
        "human_framework_convergence.csv",
    ):
        columns = {
            column.casefold() for column in pd.read_csv(root / filename).columns
        }
        assert forbidden.isdisjoint(columns)
