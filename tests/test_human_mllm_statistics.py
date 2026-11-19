from __future__ import annotations

import pandas as pd
import pytest

from biasauditfw.statistics import (
    aggregate_human_ratings,
    human_framework_convergence,
    human_judge_convergence,
    human_paired_prompt_report,
    human_participant_prompt_sensitivity,
    judge_clip_convergence,
    judge_interrater_report,
    judge_repeatability_report,
    ordinal_krippendorff_report,
    paired_simpson_prompt_report,
    pairwise_judge_report,
)


def human_ratings() -> pd.DataFrame:
    rows = []
    scores = {"i1": (1, 2), "i2": (3, 3), "i3": (5, 4)}
    for participant in ("p1", "p2", "p3", "p4"):
        for image_id, (racial, gender) in scores.items():
            rows.append(
                {
                    "participant_id": participant,
                    "imagem_id": image_id,
                    "racial_score": racial,
                    "gender_score": gender,
                }
            )
    return pd.DataFrame(rows)


def judge_scores() -> pd.DataFrame:
    rows = []
    for provider, model in (("a", "m1"), ("b", "m2")):
        for repeat in (1, 2, 3):
            for image_id, racial, gender in (("i1", 1, 2), ("i2", 3, 3), ("i3", 5, 4)):
                rows.append(
                    {
                        "provider": provider,
                        "model_id": model,
                        "imagem_id": image_id,
                        "repeat_id": repeat,
                        "eligible": True,
                        "racial_score": racial,
                        "gender_score": gender,
                    }
                )
    return pd.DataFrame(rows)


def test_human_aggregation_and_ordinal_agreement() -> None:
    human = human_ratings()
    aggregate = aggregate_human_ratings(human)
    assert "participant_id" not in aggregate
    assert aggregate.set_index("imagem_id").loc["i3", "human_racial_median"] == 5
    report = ordinal_krippendorff_report(human)
    assert report["krippendorff_alpha_ordinal"].eq(1.0).all()
    duplicate = pd.concat([human, human.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="only once"):
        aggregate_human_ratings(duplicate)
    invalid = human.copy()
    invalid.loc[0, "racial_score"] = 8
    with pytest.raises(ValueError, match="range 1-5"):
        aggregate_human_ratings(invalid)


def test_human_aggregation_includes_only_available_exploratory_scores() -> None:
    human = human_ratings()
    human["cultural_score"] = [1, 2, 3] * 4
    aggregate = aggregate_human_ratings(human)

    assert "human_cultural_median" in aggregate
    assert "human_body_median" not in aggregate
    report = ordinal_krippendorff_report(human)
    assert set(report["dimension"]) == {"racial", "gender", "cultural"}


def test_judge_repeatability_and_human_convergence() -> None:
    pointwise = judge_scores()
    repeatability = judge_repeatability_report(pointwise)
    assert repeatability["krippendorff_alpha_ordinal"].eq(1.0).all()
    convergence = human_judge_convergence(
        aggregate_human_ratings(human_ratings()),
        pointwise,
        bootstrap_iterations=50,
    )
    assert convergence["spearman_rho"].eq(1.0).all()
    assert convergence["mae"].eq(0.0).all()


def test_judge_interrater_and_clip_convergence() -> None:
    pointwise = judge_scores()
    agreement = judge_interrater_report(pointwise)
    assert agreement["n_raters"].eq(2).all()
    assert agreement["krippendorff_alpha_ordinal"].eq(1.0).all()

    images = pd.DataFrame(
        {
            "imagem_id": ["i1", "i2", "i3"],
            "clip_racial_diversity_margin": [0.1, 0.2, 0.3],
            "clip_gender_diversity_margin": [0.2, 0.3, 0.4],
        }
    )
    convergence = judge_clip_convergence(
        pointwise, images, bootstrap_iterations=50
    )
    assert set(convergence["judge_id"]) == {"a:m1", "b:m2"}
    assert convergence["n"].eq(3).all()
    assert convergence["spearman_rho"].eq(1.0).all()


def test_pairwise_report_uses_one_consistent_choice_per_pair() -> None:
    images = pd.DataFrame(
        [
            {"imagem_id": "n1", "tipo_prompt": "Neutral Prompts"},
            {"imagem_id": "i1", "tipo_prompt": "Inclusive Prompts"},
            {"imagem_id": "n2", "tipo_prompt": "Neutral Prompts"},
            {"imagem_id": "i2", "tipo_prompt": "Inclusive Prompts"},
        ]
    )
    rows = []
    for pair_id, winner in (("p1", "i1"), ("p2", "i2")):
        for order in ("neutral-first", "inclusive-first"):
            rows.append(
                {
                    "provider": "fake",
                    "model_id": "m1",
                    "dataset_id": "d1",
                    "pair_id": pair_id,
                    "order_id": order,
                    "racial_winner_id": winner,
                    "gender_winner_id": winner,
                }
            )
    report = pairwise_judge_report(pd.DataFrame(rows), images)
    assert report["position_consistency_rate"].eq(1.0).all()
    assert report["inclusive_wins"].eq(2).all()


def test_human_framework_convergence_separates_primary_and_secondary_families() -> None:
    aggregate = pd.DataFrame(
        {
            "imagem_id": [f"i{index}" for index in range(1, 7)],
            "human_racial_median": [1, 2, 3, 4, 5, 5],
            "human_gender_median": [1, 2, 3, 4, 5, 5],
        }
    )
    images = pd.DataFrame(
        {
            "imagem_id": aggregate["imagem_id"],
            "dataset_id": ["d1"] * 6,
            "num_faces": [2, 4, 3, 6, 5, 8],
            "clip_racial_diversity_margin": [0, 1, 2, 3, 4, 5],
            "clip_gender_diversity_margin": [0, 1, 2, 3, 4, 5],
        }
    )
    simpson = pd.DataFrame(
        {
            "imagem_id": aggregate["imagem_id"],
            "deepface_racial_simpson": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "deepface_gender_simpson": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        }
    )

    report = human_framework_convergence(
        aggregate, images, simpson, bootstrap_iterations=100
    )

    assert len(report) == 4
    assert set(report["analysis_family"]) == {
        "clip_primary",
        "deepface_simpson_secondary",
    }
    assert report["n"].eq(6).all()
    assert report["spearman_rho"].gt(0.9).all()
    assert report["spearman_p_value_holm"].notna().all()
    clip = report.loc[report["analysis_family"].eq("clip_primary")]
    assert clip["partial_spearman_rho_face_count"].isna().all()
    simpson_rows = report.loc[
        report["analysis_family"].eq("deepface_simpson_secondary")
    ]
    assert simpson_rows["partial_spearman_rho_face_count"].notna().all()
    assert simpson_rows[
        "partial_spearman_p_value_face_count_holm"
    ].notna().all()
    assert set(simpson_rows["partial_holm_family"]) == {
        "d1:deepface_simpson_partial"
    }


def test_human_framework_convergence_handles_missing_simpson_values() -> None:
    aggregate = pd.DataFrame(
        {
            "imagem_id": ["i1", "i2", "i3"],
            "human_racial_median": [1, 3, 5],
            "human_gender_median": [1, 3, 5],
        }
    )
    images = pd.DataFrame(
        {
            "imagem_id": ["i1", "i2", "i3"],
            "dataset_id": ["d1"] * 3,
            "num_faces": [1, 2, 3],
            "clip_racial_diversity_margin": [0.1, 0.2, 0.3],
            "clip_gender_diversity_margin": [0.1, 0.2, 0.3],
        }
    )
    simpson = pd.DataFrame(
        {
            "imagem_id": ["i1", "i2", "i3"],
            "deepface_racial_simpson": [pd.NA, 0.5, 0.8],
            "deepface_gender_simpson": [pd.NA, 0.4, 0.9],
        }
    )

    report = human_framework_convergence(
        aggregate, images, simpson, bootstrap_iterations=20
    )
    secondary = report.loc[
        report["analysis_family"].eq("deepface_simpson_secondary")
    ]
    assert secondary["n"].eq(2).all()
    assert secondary["spearman_rho"].isna().all()
    assert secondary["spearman_p_value_holm"].isna().all()


def test_human_paired_prompt_report_uses_separate_holm_families() -> None:
    aggregate = pd.DataFrame(
        {
            "imagem_id": ["n1", "i1", "n2", "i2"],
            "human_racial_median": [2, 4, 3, 4],
            "human_gender_median": [2, 5, 3, 4],
            "human_body_median": [1, 3, 2, 4],
        }
    )
    image_map = pd.DataFrame(
        {
            "imagem_id": ["n1", "i1", "n2", "i2"],
            "human_corpus_id": ["d1"] * 4,
            "human_pair_id": ["p1", "p1", "p2", "p2"],
            "human_prompt_group": ["neutral", "inclusive"] * 2,
        }
    )

    report = human_paired_prompt_report(
        aggregate, image_map, bootstrap_iterations=200, seed=7
    )

    assert set(report["dimension"]) == {"racial", "gender", "body"}
    assert report["n_pairs"].eq(2).all()
    assert set(report["holm_family"]) == {"d1:primary", "d1:exploratory"}
    assert report["p_value_holm"].notna().all()
    assert report["median_difference_ci_2_5"].notna().all()
    assert report["median_difference_ci_97_5"].notna().all()


def test_participant_sensitivity_publishes_only_aggregate_summaries() -> None:
    image_map = pd.DataFrame(
        {
            "imagem_id": ["n1", "i1", "n2", "i2"],
            "human_corpus_id": ["d1"] * 4,
            "human_pair_id": ["p1", "p1", "p2", "p2"],
            "human_prompt_group": ["neutral", "inclusive"] * 2,
        }
    )
    rows = []
    for participant, shift in (("p1", 1), ("p2", 2), ("p3", 0)):
        for image_id, base in (("n1", 2), ("i1", 2 + shift), ("n2", 3), ("i2", 3 + shift)):
            rows.append(
                {
                    "participant_id": participant,
                    "imagem_id": image_id,
                    "racial_score": base,
                    "gender_score": base,
                }
            )
    report = human_participant_prompt_sensitivity(
        pd.DataFrame(rows), image_map, bootstrap_iterations=50, seed=5
    )

    assert "participant_id" not in report.columns
    assert report["n_participants"].eq(3).all()
    assert report["n_pairs_per_participant"].eq(2).all()
    assert report["n_nonzero_participants"].eq(2).all()
    assert report["holm_family"].eq("d1:participant_sensitivity:primary").all()


def test_paired_simpson_prompt_report_normalizes_pairs_and_holm_family() -> None:
    images = pd.DataFrame(
        {
            "imagem_id": ["n1", "i1", "n2", "i2", "n3", "i3"],
            "dataset_id": ["d1"] * 6,
            "pair_id": ["p1", "p1", "p2", "p2", "p3", "p3"],
            "tipo_prompt": ["neutral", "inclusive"] * 3,
        }
    )
    simpson = pd.DataFrame(
        {
            "imagem_id": images["imagem_id"],
            "deepface_racial_simpson": [0.1, 0.4, 0.2, 0.6, 0.3, 0.7],
            "deepface_gender_simpson": [0.2, 0.5, 0.3, 0.6, 0.4, 0.8],
        }
    )

    report = paired_simpson_prompt_report(
        images, simpson, bootstrap_iterations=200, seed=11
    )

    assert set(report["dimension"]) == {"racial", "gender"}
    assert report["n_pairs"].eq(3).all()
    assert report["median_paired_difference"].gt(0).all()
    assert report["matched_rank_biserial"].eq(1.0).all()
    assert report["p_value_holm"].notna().all()
    assert set(report["holm_family"]) == {"d1:simpson_secondary"}
    assert report["difference_convention"].eq(
        "inclusive-minus-neutral"
    ).all()
