"""Image-level inference and convergent diversity measures."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.multitest import multipletests


PRIMARY_METRICS = (
    "clip_racial_diversity_margin",
    "clip_gender_diversity_margin",
)


def normalize_prompt_group(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).casefold()
    if "inclus" in text:
        return "inclusive"
    if "neutral" in text or "neutro" in text:
        return "neutral"
    return None


def require_unique_images(images: pd.DataFrame) -> None:
    if "imagem_id" not in images or images["imagem_id"].duplicated().any():
        raise ValueError("Image-level analysis requires unique imagem_id values")
    if "content_sha256" in images and images["content_sha256"].duplicated().any():
        raise ValueError("Duplicate image content would create pseudoreplication")


def analysis_images(images: pd.DataFrame) -> pd.DataFrame:
    """Return rows explicitly authorized for inferential analysis."""

    if "analysis_eligible" not in images:
        raise ValueError("analysis_eligible is required for inference")
    mask = images["analysis_eligible"].astype("boolean").fillna(False)
    eligible = images.loc[mask].copy()
    if eligible.empty:
        raise ValueError("No images are explicitly eligible for inference")
    return eligible


def inference_eligibility(images: pd.DataFrame) -> tuple[bool, str]:
    required = {"analysis_eligible", "tipo_prompt", "pair_id", *PRIMARY_METRICS}
    missing = required - set(images)
    if missing:
        return False, f"missing columns: {sorted(missing)}"
    try:
        data = analysis_images(images)
    except ValueError as exc:
        return False, str(exc)
    groups = data["tipo_prompt"].map(normalize_prompt_group)
    if not {"inclusive", "neutral"}.issubset(set(groups.dropna())):
        return False, "neutral and inclusive groups are both required"
    if data["pair_id"].isna().any():
        return False, "pair_id is required for every eligible image"
    pair_groups = pd.DataFrame({"pair_id": data["pair_id"], "group": groups})
    if pair_groups["group"].isna().any():
        return False, "tipo_prompt contains an unknown group"
    if pair_groups.duplicated(["pair_id", "group"]).any():
        return False, "pair_id and prompt group must be unique"
    if not pair_groups.groupby("pair_id")["group"].nunique().eq(2).all():
        return False, "each pair_id must contain neutral and inclusive images"
    return True, "eligible"


def rank_biserial_from_u(u_statistic: float, n_a: int, n_b: int) -> float:
    return (2.0 * u_statistic) / (n_a * n_b) - 1.0


def mann_whitney_reports(
    images: pd.DataFrame,
    metrics: Sequence[str] = PRIMARY_METRICS,
) -> pd.DataFrame:
    """Run the two primary tests and Holm-correct their p-values."""

    require_unique_images(images)
    eligible, reason = inference_eligibility(images)
    if not eligible:
        raise ValueError(f"Inference is not eligible: {reason}")
    data = analysis_images(images)
    data["grupo"] = data["tipo_prompt"].map(normalize_prompt_group)
    rows = []
    for metric in metrics:
        inclusive = data.loc[data["grupo"] == "inclusive", metric].dropna()
        neutral = data.loc[data["grupo"] == "neutral", metric].dropna()
        if inclusive.empty or neutral.empty:
            raise ValueError(f"Both groups require values for {metric}")
        statistic, p_value = mannwhitneyu(
            inclusive, neutral, alternative="two-sided"
        )
        rows.append(
            {
                "metric": metric,
                "n_inclusive": len(inclusive),
                "n_neutral": len(neutral),
                "median_inclusive": inclusive.median(),
                "iqr_inclusive": inclusive.quantile(0.75)
                - inclusive.quantile(0.25),
                "median_neutral": neutral.median(),
                "iqr_neutral": neutral.quantile(0.75) - neutral.quantile(0.25),
                "u_statistic": statistic,
                "p_value": p_value,
                "rank_biserial": rank_biserial_from_u(
                    statistic, len(inclusive), len(neutral)
                ),
            }
        )
    report = pd.DataFrame(rows)
    report["p_value_holm"] = multipletests(report["p_value"], method="holm")[1]
    report["decision_holm"] = np.where(
        report["p_value_holm"] < 0.05, "rejeita H0", "nao se rejeita H0"
    )
    return report


def wilcoxon_reports(
    images: pd.DataFrame,
    metrics: Sequence[str] = PRIMARY_METRICS,
) -> pd.DataFrame:
    """Run paired neutral-inclusive sensitivity analyses by pair_id."""

    require_unique_images(images)
    eligible, reason = inference_eligibility(images)
    if not eligible:
        raise ValueError(f"Inference is not eligible: {reason}")
    data = analysis_images(images)
    data["grupo"] = data["tipo_prompt"].map(normalize_prompt_group)
    rows = []
    for metric in metrics:
        pivot = data.pivot_table(
            index="pair_id", columns="grupo", values=metric, aggfunc="first"
        )
        pairs = pivot.dropna(subset=["inclusive", "neutral"])
        if pairs.empty:
            raise ValueError(f"No complete pairs for {metric}")
        differences = pairs["inclusive"] - pairs["neutral"]
        if np.allclose(differences, 0):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = wilcoxon(
                differences, alternative="two-sided"
            )
        rows.append(
            {
                "metric": metric,
                "n_pairs": len(pairs),
                "median_paired_difference": differences.median(),
                "n_inclusive_higher": int((differences > 0).sum()),
                "n_neutral_higher": int((differences < 0).sum()),
                "n_ties": int((differences == 0).sum()),
                "w_statistic": statistic,
                "p_value": p_value,
            }
        )
    report = pd.DataFrame(rows)
    report["p_value_holm"] = multipletests(report["p_value"], method="holm")[1]
    return report


def per_model_mann_whitney(images: pd.DataFrame) -> pd.DataFrame:
    """Run exploratory model-level tests and correct the complete family."""

    rows = []
    for model, subset in images.dropna(subset=["modelo_ia"]).groupby("modelo_ia"):
        try:
            report = mann_whitney_reports(subset).drop(
                columns=["p_value_holm", "decision_holm"]
            )
        except ValueError:
            continue
        report.insert(0, "modelo_ia", model)
        rows.extend(report.to_dict(orient="records"))
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_value_holm_family"] = multipletests(
            result["p_value"], method="holm"
        )[1]
    return result


def normalized_simpson(values: pd.Series, categories: int) -> float:
    """Return normalized Simpson diversity, or NaN for fewer than two faces."""

    valid = values.dropna()
    if len(valid) < 2:
        return float("nan")
    if categories < 2:
        raise ValueError("categories must be at least two")
    proportions = valid.value_counts(normalize=True)
    raw = 1.0 - float(np.square(proportions).sum())
    return raw / (1.0 - 1.0 / categories)


def face_diversity_by_image(faces: pd.DataFrame) -> pd.DataFrame:
    """Calculate supplementary classifier-based diversity indices."""

    rows = []
    for image_id, subset in faces.groupby("imagem_id"):
        rows.append(
            {
                "imagem_id": image_id,
                "deepface_racial_simpson": normalized_simpson(
                    subset["raca"], categories=6
                ),
                "deepface_gender_simpson": normalized_simpson(
                    subset["genero"], categories=2
                ),
            }
        )
    return pd.DataFrame(rows)


def human_interrater_report(human: pd.DataFrame) -> pd.DataFrame:
    dimensions = (
        ("racial_diversity", "R1_Racial_Diversity", "R2_Racial_Diversity"),
        (
            "gender_representation",
            "R1_Gender_Representation",
            "R2_Gender_Representation",
        ),
    )
    rows = []
    for dimension, r1, r2 in dimensions:
        valid = human[[r1, r2]].apply(pd.to_numeric, errors="coerce").dropna()
        rows.append(
            {
                "dimension": dimension,
                "n": len(valid),
                "weighted_kappa": cohen_kappa_score(
                    valid[r1], valid[r2], weights="quadratic"
                ),
            }
        )
    return pd.DataFrame(rows)


def human_clip_correlations(
    human: pd.DataFrame, images: pd.DataFrame
) -> pd.DataFrame:
    """Match explicit source IDs and correlate corresponding dimensions."""

    if "source_image_id" not in images:
        raise ValueError("source_image_id is required for human validation")
    merged = human.merge(
        images[
            [
                "source_image_id",
                "clip_racial_diversity_margin",
                "clip_gender_diversity_margin",
            ]
        ],
        left_on="Image_id",
        right_on="source_image_id",
        how="inner",
        validate="one_to_one",
    )
    mappings = (
        (
            "racial_diversity",
            ["R1_Racial_Diversity", "R2_Racial_Diversity"],
            "clip_racial_diversity_margin",
        ),
        (
            "gender_representation",
            ["R1_Gender_Representation", "R2_Gender_Representation"],
            "clip_gender_diversity_margin",
        ),
    )
    rows = []
    for dimension, human_columns, clip_column in mappings:
        human_mean = merged[human_columns].apply(
            pd.to_numeric, errors="coerce"
        ).mean(axis=1)
        valid = pd.DataFrame(
            {"human": human_mean, "clip": merged[clip_column]}
        ).dropna()
        rho, p_value = spearmanr(valid["human"], valid["clip"])
        rows.append(
            {
                "dimension": dimension,
                "n": len(valid),
                "spearman_rho": rho,
                "p_value": p_value,
            }
        )
    report = pd.DataFrame(rows)
    report["p_value_holm"] = multipletests(report["p_value"], method="holm")[1]
    return report
