from __future__ import annotations

import pandas as pd
import pytest

from biasauditfw.sampling import build_unesp_form_orders, select_human_validation_sample


def test_hybrid_sample_has_24_images_from_two_datasets() -> None:
    images, scores = [], []
    models = ("DALL-E", "Midjourney", "Whisk", "NanoBanana")
    for dataset in ("original", "replication"):
        for pair_index in range(8):
            model = models[pair_index % 4]
            for group in ("Neutral Prompts", "Inclusive Prompts"):
                image_id = f"{dataset}-{pair_index}-{group[0]}"
                images.append(
                    {
                        "dataset_id": dataset,
                        "pair_id": f"p{pair_index}",
                        "tipo_prompt": group,
                        "modelo_ia": model,
                        "imagem_id": image_id,
                        "analysis_eligible": True,
                    }
                )
                for judge_index in range(3):
                    scores.append(
                        {
                            "imagem_id": image_id,
                            "provider": f"provider-{judge_index}",
                            "model_id": f"model-{judge_index}",
                            "eligible": True,
                            "racial_score": 1 + (pair_index + judge_index) % 5,
                            "gender_score": 1 + (pair_index * 2 + judge_index) % 5,
                        }
                    )
    sample = select_human_validation_sample(
        pd.DataFrame(images), pd.DataFrame(scores), seed=7
    )
    assert len(sample) == 24
    assert sample["imagem_id"].nunique() == 24
    assert sample.groupby("dataset_id").size().eq(12).all()
    assert (sample["selection_component"] == "targeted").sum() == 8
    assert (sample["selection_component"] == "stratified_random").sum() == 16
    assert sample.groupby(["dataset_id", "pair_id"])["tipo_prompt"].nunique().eq(2).all()


def test_unesp_orders_are_balanced_and_hide_pair_counterparts() -> None:
    rows = []
    for dataset in ("original", "vitor"):
        for model in ("dalle", "gemini", "midjourney", "whisk"):
            for prompt_id in ("P2", "P4", "P6"):
                pair_id = f"{dataset}-{model}-{prompt_id}"
                for group in ("Neutral Prompts", "Inclusive Prompts"):
                    image_id = f"{pair_id}-{group[0]}"
                    rows.append(
                        {
                            "dataset_id": dataset,
                            "pair_id": pair_id,
                            "prompt_id": prompt_id,
                            "tipo_prompt": group,
                            "modelo_ia": model,
                            "imagem_id": image_id,
                            "caminho_relativo": f"{image_id}.png",
                        }
                    )
    orders = build_unesp_form_orders(pd.DataFrame(rows), seed=7)
    assert set(orders) == {"A1", "B1", "A2", "B2"}
    assert all(len(order) == 24 for order in orders.values())
    assert set(orders["A1"]["imagem_id"]) == set(orders["B1"]["imagem_id"])
    assert set(orders["A2"]["imagem_id"]) == set(orders["B2"]["imagem_id"])
    assert set(orders["A1"]["imagem_id"]).isdisjoint(orders["A2"]["imagem_id"])
    for order in orders.values():
        assert order["pair_id"].is_unique
        assert order["prompt_group"].value_counts().to_dict() == {
            "neutral": 12,
            "inclusive": 12,
        }


def test_unesp_orders_reject_incomplete_selected_pair() -> None:
    rows = []
    for dataset in ("original", "vitor"):
        for model in ("dalle", "gemini", "midjourney", "whisk"):
            for prompt_id in ("P2", "P4", "P6"):
                groups = ("Neutral Prompts", "Inclusive Prompts")
                if dataset == "vitor" and model == "whisk" and prompt_id == "P4":
                    groups = ("Neutral Prompts",)
                for group in groups:
                    pair_id = f"{dataset}-{model}-{prompt_id}"
                    rows.append(
                        {
                            "dataset_id": dataset,
                            "pair_id": pair_id,
                            "prompt_id": prompt_id,
                            "tipo_prompt": group,
                            "modelo_ia": model,
                            "imagem_id": f"{pair_id}-{group[0]}",
                            "caminho_relativo": f"{pair_id}-{group[0]}.png",
                        }
                    )
    with pytest.raises(ValueError, match="Incomplete selected pair"):
        build_unesp_form_orders(pd.DataFrame(rows))


def test_unesp_orders_can_partition_two_unpaired_neutral_images() -> None:
    rows = []
    for dataset in ("original", "vitor"):
        for model in ("dalle", "gemini", "midjourney", "whisk"):
            for prompt_id in ("P2", "P4", "P6"):
                groups = ("Neutral Prompts", "Inclusive Prompts")
                if (
                    dataset == "vitor"
                    and model in {"midjourney", "whisk"}
                    and prompt_id == "P4"
                ):
                    groups = ("Neutral Prompts",)
                for group in groups:
                    pair_id = f"{dataset}-{model}-{prompt_id}"
                    rows.append(
                        {
                            "dataset_id": dataset,
                            "pair_id": pair_id,
                            "prompt_id": prompt_id,
                            "tipo_prompt": group,
                            "modelo_ia": model,
                            "imagem_id": f"{pair_id}-{group[0]}",
                            "caminho_relativo": f"{pair_id}-{group[0]}.png",
                        }
                    )
    orders = build_unesp_form_orders(
        pd.DataFrame(rows), seed=7, allow_unpaired=True
    )
    assert all(len(order) == 23 for order in orders.values())
    assert set(orders["A1"]["imagem_id"]).isdisjoint(orders["A2"]["imagem_id"])
    assert set(orders["A1"]["imagem_id"]) | set(orders["A2"]["imagem_id"]) == {
        row["imagem_id"] for row in rows
    }
    for order in orders.values():
        assert order["prompt_group"].value_counts().to_dict() == {
            "neutral": 12,
            "inclusive": 11,
        }
