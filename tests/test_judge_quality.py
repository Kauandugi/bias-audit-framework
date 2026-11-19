from __future__ import annotations

import pandas as pd

from biasauditfw.judges.quality import (
    evaluate_semantic_gate,
    select_semantic_pilot,
)


def _pilot_inventory() -> pd.DataFrame:
    rows = []
    specifications = [
        ("model-a", "p1", 8, 7),
        ("model-a", "p2", 4, 4),
        ("model-b", "p3", 6, 5),
        ("model-b", "p4", 3, 2),
    ]
    for model, pair_id, neutral_faces, inclusive_faces in specifications:
        for group, faces in (
            ("Neutral Prompts", neutral_faces),
            ("Inclusive Prompts", inclusive_faces),
        ):
            rows.append(
                {
                    "dataset_id": "original-64",
                    "pair_id": pair_id,
                    "tipo_prompt": group,
                    "imagem_id": f"{pair_id}-{group[0].lower()}",
                    "modelo_ia": model,
                    "num_faces": faces,
                    "analysis_eligible": True,
                }
            )
    return pd.DataFrame(rows)


def test_semantic_pilot_selection_preserves_complete_pairs_and_models() -> None:
    selected = select_semantic_pilot(_pilot_inventory(), pair_count=2)

    assert len(selected) == 4
    assert set(selected["pair_id"]) == {"p1", "p3"}
    assert set(selected["modelo_ia"]) == {"model-a", "model-b"}
    assert selected.groupby("pair_id")["tipo_prompt"].nunique().eq(2).all()


def test_semantic_pilot_prefers_clip_contrast_within_each_model() -> None:
    inventory = _pilot_inventory()
    inventory["clip_racial_diversity_margin"] = 0.0
    inventory["clip_gender_diversity_margin"] = 0.0
    inventory.loc[
        (inventory["pair_id"] == "p2")
        & (inventory["tipo_prompt"] == "Inclusive Prompts"),
        "clip_racial_diversity_margin",
    ] = 1.0
    inventory.loc[
        (inventory["pair_id"] == "p4")
        & (inventory["tipo_prompt"] == "Inclusive Prompts"),
        "clip_gender_diversity_margin",
    ] = 0.8

    selected = select_semantic_pilot(inventory, pair_count=2)

    assert set(selected["pair_id"]) == {"p2", "p4"}


def _passing_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    pointwise = pd.DataFrame(
        {
            "dataset_id": ["original-64"] * 8,
            "imagem_id": [f"image-{index}" for index in range(8)],
            "eligible": [True, True, True, True, True, True, False, False],
            "racial_score": [1, 2, 3, 4, 2, 3, None, None],
            "gender_score": [2, 3, 4, 5, 3, 2, None, None],
        }
    )
    semantic_choices = {
        "p1": ("inclusive", "neutral"),
        "p2": ("tie", "tie"),
        "p3": ("neutral", "inclusive"),
        "p4": ("inclusive", "tie"),
    }
    rows = []
    for pair_id, (racial, gender) in semantic_choices.items():
        for order in ("neutral-first", "inclusive-first"):
            def positional(value: str) -> str:
                if value in {"tie", "not_evaluable"}:
                    return value
                if order == "neutral-first":
                    return "A" if value == "neutral" else "B"
                return "B" if value == "neutral" else "A"

            rows.append(
                {
                    "dataset_id": "original-64",
                    "pair_id": pair_id,
                    "order_id": order,
                    "racial_choice": positional(racial),
                    "gender_choice": positional(gender),
                }
            )
    return pointwise, pd.DataFrame(rows)


def test_semantic_gate_accepts_varied_order_consistent_pilot() -> None:
    report = evaluate_semantic_gate(*_passing_outputs())

    assert report["passed"] is True
    assert report["metrics"]["complete_pairwise_pairs"] == 4
    assert report["metrics"]["racial_order_consistency_rate"] == 1.0
    assert all(report["checks"].values())


def test_semantic_gate_rejects_degenerate_qwen_pattern() -> None:
    pointwise, pairwise = _passing_outputs()
    pointwise["eligible"] = [True] + [False] * 7
    pointwise.loc[~pointwise["eligible"], ["racial_score", "gender_score"]] = None
    pointwise.loc[pointwise["eligible"], ["racial_score", "gender_score"]] = 1
    pairwise["racial_choice"] = "not_evaluable"
    pairwise["gender_choice"] = "tie"

    report = evaluate_semantic_gate(pointwise, pairwise)

    assert report["passed"] is False
    assert report["checks"]["acceptable_abstention_rate"] is False
    assert report["checks"]["acceptable_racial_not_evaluable_rate"] is False
    assert report["checks"]["pairwise_outcomes_are_not_constant"] is False
