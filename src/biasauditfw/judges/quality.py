"""Engineering quality gate for a small multimodal-judge pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from ..statistics import normalize_prompt_group


@dataclass(frozen=True)
class SemanticGateConfig:
    """Conservative checks that reject unusable, constant judge behavior.

    This is a preflight gate, not evidence that a judge is accurate. Academic
    validity still depends on comparison with the perceptual human reference.
    """

    min_pointwise_records: int = 8
    min_pairwise_pairs: int = 4
    max_pointwise_abstention_rate: float = 0.50
    max_pairwise_not_evaluable_rate: float = 0.50
    min_unique_scores_per_dimension: int = 2
    min_joint_pairwise_outcomes: int = 2
    min_order_consistency_rate: float = 0.50

    def __post_init__(self) -> None:
        if self.min_pointwise_records < 1 or self.min_pairwise_pairs < 1:
            raise ValueError("Semantic gate record minimums must be positive")
        for field in (
            "max_pointwise_abstention_rate",
            "max_pairwise_not_evaluable_rate",
            "min_order_consistency_rate",
        ):
            value = float(getattr(self, field))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be in [0, 1]")
        if self.min_unique_scores_per_dimension < 1:
            raise ValueError("min_unique_scores_per_dimension must be positive")
        if self.min_joint_pairwise_outcomes < 1:
            raise ValueError("min_joint_pairwise_outcomes must be positive")


def select_semantic_pilot(
    images: pd.DataFrame,
    pair_count: int = 4,
    min_faces_per_image: int = 2,
) -> pd.DataFrame:
    """Select complete, detector-screened pairs with generator coverage.

    Face counts are used only to avoid an uninformative smoke sample. When the
    two CLIP margins are available, within-pair semantic contrast determines
    the ranking inside each generator. Neither signal is treated as human
    Ground Truth or as proof that the detector or CLIP is correct.
    """

    if pair_count < 1 or min_faces_per_image < 0:
        raise ValueError("pair_count must be positive and min_faces non-negative")
    required = {
        "dataset_id",
        "pair_id",
        "tipo_prompt",
        "imagem_id",
        "num_faces",
    }
    missing = required - set(images)
    if missing:
        raise ValueError(f"Pilot inventory is missing columns: {sorted(missing)}")
    if images["imagem_id"].duplicated().any():
        raise ValueError("Pilot selection requires unique imagem_id values")

    data = images.copy()
    if "analysis_eligible" in data:
        eligible = data["analysis_eligible"].astype(str).str.casefold().isin(
            {"true", "1", "yes"}
        )
        data = data.loc[eligible].copy()
    data["_prompt_group"] = data["tipo_prompt"].map(normalize_prompt_group)
    data["_model"] = (
        data["modelo_ia"].fillna("unknown").astype(str)
        if "modelo_ia" in data
        else "unknown"
    )
    data["num_faces"] = pd.to_numeric(data["num_faces"], errors="raise")

    margin_columns = {
        "clip_racial_diversity_margin",
        "clip_gender_diversity_margin",
    }
    has_clip_margins = margin_columns.issubset(data.columns)
    candidates: list[dict[str, Any]] = []
    group_columns = ["dataset_id", "pair_id"]
    for (dataset_id, pair_id), group in data.groupby(group_columns, sort=True):
        if len(group) != 2 or set(group["_prompt_group"]) != {
            "neutral",
            "inclusive",
        }:
            continue
        model_values = sorted(set(group["_model"]))
        if len(model_values) != 1:
            continue
        minimum = int(group["num_faces"].min())
        if minimum < min_faces_per_image:
            continue
        semantic_contrast = 0.0
        if has_clip_margins:
            semantic_contrast = float(
                sum(
                    abs(float(group[column].iloc[0]) - float(group[column].iloc[1]))
                    for column in sorted(margin_columns)
                )
            )
        candidates.append(
            {
                "dataset_id": str(dataset_id),
                "pair_id": str(pair_id),
                "modelo_ia": model_values[0],
                "min_faces": minimum,
                "total_faces": int(group["num_faces"].sum()),
                "semantic_contrast": semantic_contrast,
            }
        )

    if len(candidates) < pair_count:
        raise ValueError(
            f"Only {len(candidates)} complete pairs have at least "
            f"{min_faces_per_image} detected faces per image; {pair_count} required"
        )
    ranked = sorted(
        candidates,
        key=lambda row: (
            -row["semantic_contrast"],
            -row["min_faces"],
            -row["total_faces"],
            row["modelo_ia"],
            row["dataset_id"],
            row["pair_id"],
        ),
    )

    chosen: list[dict[str, Any]] = []
    used_keys: set[tuple[str, str]] = set()
    for model in sorted({row["modelo_ia"] for row in ranked}):
        best = next(row for row in ranked if row["modelo_ia"] == model)
        key = (best["dataset_id"], best["pair_id"])
        chosen.append(best)
        used_keys.add(key)
        if len(chosen) == pair_count:
            break
    if len(chosen) < pair_count:
        for row in ranked:
            key = (row["dataset_id"], row["pair_id"])
            if key in used_keys:
                continue
            chosen.append(row)
            used_keys.add(key)
            if len(chosen) == pair_count:
                break

    rank_by_key = {
        (row["dataset_id"], row["pair_id"]): rank
        for rank, row in enumerate(chosen, start=1)
    }
    keys = pd.MultiIndex.from_tuples(rank_by_key, names=group_columns)
    selected = data.set_index(group_columns).loc[keys].reset_index()
    selected["pilot_pair_rank"] = [
        rank_by_key[(str(dataset_id), str(pair_id))]
        for dataset_id, pair_id in selected[group_columns].itertuples(index=False)
    ]
    return selected.drop(columns=["_prompt_group", "_model"]).sort_values(
        ["pilot_pair_rank", "tipo_prompt", "imagem_id"], ignore_index=True
    )


def _semantic_choice(choice: object, order_id: object) -> str:
    value = str(choice)
    order = str(order_id)
    if value in {"tie", "not_evaluable"}:
        return value
    if value not in {"A", "B"}:
        raise ValueError(f"Unknown pairwise choice: {value}")
    if order == "neutral-first":
        return "neutral" if value == "A" else "inclusive"
    if order == "inclusive-first":
        return "inclusive" if value == "A" else "neutral"
    raise ValueError(f"Unknown order_id: {order}")


def _order_consistency(pairwise: pd.DataFrame, dimension: str) -> tuple[float, int]:
    comparable = 0
    agreements = 0
    semantic_column = f"_{dimension}_semantic"
    for _, group in pairwise.groupby(["dataset_id", "pair_id"], sort=True):
        if set(group["order_id"]) != {"neutral-first", "inclusive-first"}:
            continue
        values = set(group[semantic_column])
        if "not_evaluable" in values:
            continue
        comparable += 1
        agreements += int(len(values) == 1)
    return (agreements / comparable if comparable else 0.0), comparable


def evaluate_semantic_gate(
    pointwise: pd.DataFrame,
    pairwise: pd.DataFrame,
    config: SemanticGateConfig | None = None,
) -> dict[str, Any]:
    """Evaluate a single provider pilot and return a serializable report."""

    config = config or SemanticGateConfig()
    point_required = {
        "dataset_id",
        "imagem_id",
        "eligible",
        "racial_score",
        "gender_score",
    }
    pair_required = {
        "dataset_id",
        "pair_id",
        "order_id",
        "racial_choice",
        "gender_choice",
    }
    if missing := point_required - set(pointwise):
        raise ValueError(f"Pointwise pilot is missing columns: {sorted(missing)}")
    if missing := pair_required - set(pairwise):
        raise ValueError(f"Pairwise pilot is missing columns: {sorted(missing)}")
    if pointwise["imagem_id"].duplicated().any():
        raise ValueError("Semantic pilot requires one pointwise row per image")
    if pairwise.duplicated(["dataset_id", "pair_id", "order_id"]).any():
        raise ValueError("Semantic pilot contains duplicate pair/order rows")

    point = pointwise.copy()
    eligible = point["eligible"].astype(str).str.casefold().isin({"true", "1"})
    abstention_rate = float((~eligible).mean()) if len(point) else 1.0
    eligible_point = point.loc[eligible]
    unique_racial_scores = int(eligible_point["racial_score"].dropna().nunique())
    unique_gender_scores = int(eligible_point["gender_score"].dropna().nunique())

    pair = pairwise.copy()
    for dimension in ("racial", "gender"):
        pair[f"_{dimension}_semantic"] = [
            _semantic_choice(choice, order)
            for choice, order in pair[[f"{dimension}_choice", "order_id"]].itertuples(
                index=False
            )
        ]
    complete_pairs = int(
        sum(
            set(group["order_id"]) == {"neutral-first", "inclusive-first"}
            for _, group in pair.groupby(["dataset_id", "pair_id"], sort=True)
        )
    )
    racial_not_evaluable = float(
        (pair["racial_choice"] == "not_evaluable").mean()
    ) if len(pair) else 1.0
    gender_not_evaluable = float(
        (pair["gender_choice"] == "not_evaluable").mean()
    ) if len(pair) else 1.0
    racial_consistency, racial_comparable = _order_consistency(pair, "racial")
    gender_consistency, gender_comparable = _order_consistency(pair, "gender")
    joint_outcomes = int(
        pair[["_racial_semantic", "_gender_semantic"]].drop_duplicates().shape[0]
    )

    metrics = {
        "pointwise_records": int(len(point)),
        "pointwise_eligible": int(eligible.sum()),
        "pointwise_abstention_rate": abstention_rate,
        "unique_racial_scores": unique_racial_scores,
        "unique_gender_scores": unique_gender_scores,
        "pairwise_records": int(len(pair)),
        "complete_pairwise_pairs": complete_pairs,
        "racial_not_evaluable_rate": racial_not_evaluable,
        "gender_not_evaluable_rate": gender_not_evaluable,
        "joint_pairwise_outcomes": joint_outcomes,
        "racial_order_consistency_rate": racial_consistency,
        "racial_order_comparable_pairs": racial_comparable,
        "gender_order_consistency_rate": gender_consistency,
        "gender_order_comparable_pairs": gender_comparable,
    }
    checks = {
        "enough_pointwise_records": len(point) >= config.min_pointwise_records,
        "acceptable_abstention_rate": (
            abstention_rate <= config.max_pointwise_abstention_rate
        ),
        "racial_scores_are_not_constant": (
            unique_racial_scores >= config.min_unique_scores_per_dimension
        ),
        "gender_scores_are_not_constant": (
            unique_gender_scores >= config.min_unique_scores_per_dimension
        ),
        "enough_complete_pairs": complete_pairs >= config.min_pairwise_pairs,
        "acceptable_racial_not_evaluable_rate": (
            racial_not_evaluable <= config.max_pairwise_not_evaluable_rate
        ),
        "acceptable_gender_not_evaluable_rate": (
            gender_not_evaluable <= config.max_pairwise_not_evaluable_rate
        ),
        "pairwise_outcomes_are_not_constant": (
            joint_outcomes >= config.min_joint_pairwise_outcomes
        ),
        "racial_order_is_consistent": (
            racial_comparable > 0
            and racial_consistency >= config.min_order_consistency_rate
        ),
        "gender_order_is_consistent": (
            gender_comparable > 0
            and gender_consistency >= config.min_order_consistency_rate
        ),
    }
    return {
        "gate_version": "semantic-pilot-v1",
        "purpose": "engineering degeneration screen; not an accuracy claim",
        "config": asdict(config),
        "metrics": metrics,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }
