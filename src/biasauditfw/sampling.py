"""Deterministic sampling for a perceptual human-validation form."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .statistics import normalize_prompt_group


def _complete_pairs(images: pd.DataFrame) -> pd.DataFrame:
    data = images.copy()
    if "analysis_eligible" in data:
        data = data.loc[data["analysis_eligible"].astype("boolean").fillna(False)]
    data["prompt_group"] = data["tipo_prompt"].map(normalize_prompt_group)
    rows = []
    for (dataset_id, pair_id), group in data.groupby(["dataset_id", "pair_id"]):
        if len(group) != 2 or set(group["prompt_group"]) != {"neutral", "inclusive"}:
            continue
        models = group["modelo_ia"].dropna().astype(str).unique()
        if len(models) != 1:
            continue
        rows.append(
            {
                "dataset_id": dataset_id,
                "pair_id": pair_id,
                "modelo_ia": models[0],
                "imagem_ids": tuple(group["imagem_id"].astype(str)),
            }
        )
    return pd.DataFrame(rows)


def _pair_priority(pointwise: pd.DataFrame) -> pd.Series:
    required = {"imagem_id", "provider", "model_id", "eligible", "racial_score", "gender_score"}
    missing = required - set(pointwise)
    if missing:
        raise ValueError(f"Judge table is missing columns: {sorted(missing)}")
    data = pointwise.loc[pointwise["eligible"].astype(bool)].copy()
    data["judge_id"] = data["provider"].astype(str) + ":" + data["model_id"].astype(str)
    per_judge = (
        data.groupby(["imagem_id", "judge_id"])[["racial_score", "gender_score"]]
        .median()
        .reset_index()
    )
    grouped = per_judge.groupby("imagem_id")
    disagreement = grouped[["racial_score", "gender_score"]].std(ddof=0).mean(axis=1)
    medians = grouped[["racial_score", "gender_score"]].median()
    extremeness = medians.sub(3).abs().mean(axis=1) / 2.0
    return disagreement.add(extremeness, fill_value=0.0)


def select_human_validation_sample(
    images: pd.DataFrame,
    pointwise: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Select 24 images: 16 stratified-random and 8 targeted, in complete pairs."""

    required = {"dataset_id", "pair_id", "tipo_prompt", "modelo_ia", "imagem_id"}
    missing = required - set(images)
    if missing:
        raise ValueError(f"Image inventory is missing columns: {sorted(missing)}")
    pairs = _complete_pairs(images)
    datasets = sorted(pairs["dataset_id"].astype(str).unique()) if not pairs.empty else []
    if len(datasets) != 2:
        raise ValueError("Human sampling requires exactly two datasets with complete pairs")
    if any(len(pairs[pairs["dataset_id"].astype(str) == dataset]) < 6 for dataset in datasets):
        raise ValueError("Each dataset requires at least six complete pairs")

    priority = _pair_priority(pointwise)
    pairs["target_priority"] = pairs["imagem_ids"].map(
        lambda ids: max((priority.get(image_id, 0.0) for image_id in ids), default=0.0)
    )
    rng = np.random.default_rng(seed)
    pairs["random_key"] = rng.random(len(pairs))
    selected: list[dict[str, object]] = []
    model_counts: Counter[str] = Counter()

    for dataset in datasets:
        candidates = pairs[pairs["dataset_id"].astype(str) == dataset].copy()
        targeted = candidates.sort_values(
            ["target_priority", "random_key"], ascending=[False, True]
        ).head(2)
        for row in targeted.to_dict(orient="records"):
            row["selection_component"] = "targeted"
            selected.append(row)
            model_counts[str(row["modelo_ia"])] += 1

    selected_keys = {(str(row["dataset_id"]), str(row["pair_id"])) for row in selected}
    for dataset in datasets:
        candidates = pairs[
            (pairs["dataset_id"].astype(str) == dataset)
            & ~pairs.apply(
                lambda row: (str(row["dataset_id"]), str(row["pair_id"])) in selected_keys,
                axis=1,
            )
        ].copy()
        for _ in range(4):
            candidates["model_count"] = candidates["modelo_ia"].astype(str).map(
                lambda value: model_counts[value]
            )
            choice = candidates.sort_values(
                ["model_count", "random_key", "modelo_ia"],
                ascending=[True, True, True],
            ).iloc[0]
            row = choice.to_dict()
            row["selection_component"] = "stratified_random"
            selected.append(row)
            model_counts[str(row["modelo_ia"])] += 1
            candidates = candidates.drop(index=choice.name)

    selection = pd.DataFrame(selected)
    pair_metadata = selection[["dataset_id", "pair_id", "selection_component"]]
    result = images.merge(pair_metadata, on=["dataset_id", "pair_id"], how="inner")
    if len(result) != 24 or result["imagem_id"].nunique() != 24:
        raise AssertionError("Hybrid sampling must produce exactly 24 unique images")
    return result.sort_values(
        ["dataset_id", "selection_component", "modelo_ia", "pair_id", "tipo_prompt"]
    ).reset_index(drop=True)


def build_unesp_form_orders(
    images: pd.DataFrame,
    prompt_ids: tuple[str, ...] = ("P2", "P4", "P6"),
    seed: int = 42,
    allow_unpaired: bool = False,
) -> dict[str, pd.DataFrame]:
    """Create four blinded forms without showing both sides of a complete pair."""

    required = {
        "dataset_id",
        "pair_id",
        "prompt_id",
        "tipo_prompt",
        "modelo_ia",
        "imagem_id",
        "caminho_relativo",
    }
    missing = required - set(images)
    if missing:
        raise ValueError(f"Image inventory is missing columns: {sorted(missing)}")
    data = images.copy()
    data["prompt_id"] = data["prompt_id"].astype(str).str.upper()
    data = data.loc[data["prompt_id"].isin(prompt_ids)].copy()
    data["prompt_group"] = data["tipo_prompt"].map(normalize_prompt_group)
    if data["prompt_group"].isna().any():
        raise ValueError("Selected images contain an unknown prompt group")

    datasets = sorted(data["dataset_id"].astype(str).unique())
    models_by_dataset = data.groupby("dataset_id")["modelo_ia"].nunique()
    if len(datasets) != 2 or not models_by_dataset.eq(4).all():
        raise ValueError(
            "UNESP forms require exactly two datasets and four models per dataset"
        )

    complete_pairs: list[pd.DataFrame] = []
    unpaired_rows: list[pd.Series] = []
    for (dataset_id, pair_id), group in data.groupby(
        ["dataset_id", "pair_id"], sort=True
    ):
        is_complete = len(group) == 2 and set(group["prompt_group"]) == {
            "neutral",
            "inclusive",
        }
        if is_complete:
            complete_pairs.append(group)
            continue
        if allow_unpaired and len(group) == 1:
            unpaired_rows.append(group.iloc[0])
            continue
        if not is_complete:
            raise ValueError(
                f"Incomplete selected pair: dataset={dataset_id}, pair_id={pair_id}"
            )
    expected_pairs = int(models_by_dataset.sum()) * len(prompt_ids)
    missing_images = expected_pairs * 2 - len(data)
    if missing_images < 0 or len(complete_pairs) + len(unpaired_rows) != expected_pairs:
        raise ValueError("Selected images do not map one-to-one to the expected pairs")
    if not allow_unpaired and (
        len(complete_pairs) != expected_pairs or len(data) != expected_pairs * 2
    ):
        raise ValueError(
            f"Expected {expected_pairs} complete pairs and {expected_pairs * 2} images; "
            f"found {len(complete_pairs)} pairs and {len(data)} images"
        )
    if allow_unpaired and missing_images != len(unpaired_rows):
        raise ValueError("Every missing pair side must have exactly one unpaired image")

    pair_table = (
        pd.concat(complete_pairs, ignore_index=True)[
            ["dataset_id", "modelo_ia", "prompt_id", "pair_id"]
        ]
        .drop_duplicates()
        .sort_values(["dataset_id", "modelo_ia", "prompt_id", "pair_id"])
        .reset_index(drop=True)
    )
    panel_rows: dict[int, list[pd.Series]] = {1: [], 2: []}
    for index, pair in pair_table.iterrows():
        group = data.loc[
            (data["dataset_id"] == pair["dataset_id"])
            & (data["pair_id"] == pair["pair_id"])
        ]
        neutral = group.loc[group["prompt_group"] == "neutral"].iloc[0]
        inclusive = group.loc[group["prompt_group"] == "inclusive"].iloc[0]
        if index % 2 == 0:
            panel_rows[1].append(neutral)
            panel_rows[2].append(inclusive)
        else:
            panel_rows[1].append(inclusive)
            panel_rows[2].append(neutral)

    for index, row in enumerate(
        sorted(
            unpaired_rows,
            key=lambda value: (
                str(value["dataset_id"]),
                str(value["modelo_ia"]),
                str(value["prompt_id"]),
                str(value["imagem_id"]),
            ),
        )
    ):
        panel_rows[1 + index % 2].append(row)

    orders: dict[str, pd.DataFrame] = {}
    for panel_number, rows in panel_rows.items():
        panel = pd.DataFrame(rows).reset_index(drop=True)
        order_a = panel.sample(frac=1, random_state=seed + panel_number).reset_index(
            drop=True
        )
        order_b = order_a.iloc[::-1].reset_index(drop=True)
        for order_name, order in (("A", order_a), ("B", order_b)):
            form_code = f"{order_name}{panel_number}"
            result = order.copy()
            result.insert(0, "form_code", form_code)
            result.insert(1, "form_position", range(1, len(result) + 1))
            orders[form_code] = result

    panel_1 = set(orders["A1"]["imagem_id"])
    panel_2 = set(orders["A2"]["imagem_id"])
    if panel_1 & panel_2 or panel_1 | panel_2 != set(data["imagem_id"]):
        raise AssertionError("UNESP panels must partition the selected images")
    if abs(len(panel_1) - len(panel_2)) > 1:
        raise AssertionError("UNESP panels must have balanced lengths")
    return orders
