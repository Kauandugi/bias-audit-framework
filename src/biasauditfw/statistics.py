"""Image-level inference and convergent diversity measures."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import krippendorff
from scipy.stats import (
    binomtest,
    mannwhitneyu,
    pearsonr,
    rankdata,
    spearmanr,
    wilcoxon,
)
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.multitest import multipletests


DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_RANDOM_SEED = 42


def _paired_median_bootstrap(
    differences: pd.Series,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[float, float]:
    """Return a percentile interval for the median of paired differences."""

    numeric = pd.to_numeric(differences, errors="coerce").dropna()
    if numeric.empty:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    values = numeric.to_numpy(dtype=float)
    estimates = np.empty(iterations, dtype=float)
    for index in range(iterations):
        estimates[index] = np.median(
            values[rng.integers(0, len(values), len(values))]
        )
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


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
    """Run the two unpaired sensitivity tests with Holm correction."""

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
        combined = pd.concat([inclusive, neutral], ignore_index=True)
        if combined.nunique(dropna=True) == 1:
            # Some SciPy releases return NaN when every rank is tied. In this
            # degenerate null case, U is centered and the two-sided p-value is 1.
            statistic = len(inclusive) * len(neutral) / 2.0
            p_value = 1.0
        else:
            statistic, p_value = mannwhitneyu(
                inclusive, neutral, alternative="two-sided"
            )
        if not np.isfinite(statistic) or not np.isfinite(p_value):
            raise ValueError(f"Mann-Whitney is undefined for {metric}")
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
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Run the primary paired neutral-inclusive analyses by pair_id."""

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
        ci_low, ci_high = _paired_median_bootstrap(
            differences,
            iterations=bootstrap_iterations,
            seed=seed + len(rows),
        )
        if np.allclose(differences, 0):
            statistic, p_value = 0.0, 1.0
            matched_rank_biserial = 0.0
        else:
            statistic, p_value = wilcoxon(
                differences, alternative="two-sided"
            )
            nonzero = differences[differences != 0]
            ranks = rankdata(nonzero.abs())
            positive = float(ranks[nonzero.to_numpy() > 0].sum())
            negative = float(ranks[nonzero.to_numpy() < 0].sum())
            matched_rank_biserial = (positive - negative) / (positive + negative)
        rows.append(
            {
                "metric": metric,
                "n_pairs": len(pairs),
                "median_paired_difference": differences.median(),
                "median_difference_ci_2_5": ci_low,
                "median_difference_ci_97_5": ci_high,
                "n_inclusive_higher": int((differences > 0).sum()),
                "n_neutral_higher": int((differences < 0).sum()),
                "n_ties": int((differences == 0).sum()),
                "w_statistic": statistic,
                "p_value": p_value,
                "matched_rank_biserial": matched_rank_biserial,
            }
        )
    report = pd.DataFrame(rows)
    report["p_value_holm"] = multipletests(report["p_value"], method="holm")[1]
    return report


PRIMARY_HUMAN_SCORE_COLUMNS = {
    "racial": "racial_score",
    "gender": "gender_score",
}

EXPLORATORY_HUMAN_SCORE_COLUMNS = {
    "cultural": "cultural_score",
    "body": "body_score",
    "quality": "quality_score",
}

HUMAN_SCORE_COLUMNS = {
    **PRIMARY_HUMAN_SCORE_COLUMNS,
    **EXPLORATORY_HUMAN_SCORE_COLUMNS,
}


def available_human_score_columns(human: pd.DataFrame) -> dict[str, str]:
    """Return primary scores plus exploratory scores present in a rating table."""

    missing = set(PRIMARY_HUMAN_SCORE_COLUMNS.values()) - set(human)
    if missing:
        raise ValueError(f"Human ratings are missing columns: {sorted(missing)}")
    return {
        dimension: column
        for dimension, column in HUMAN_SCORE_COLUMNS.items()
        if column in human
    }


def _validated_ordinal_scores(values: pd.Series, field: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = numeric.notna() & (~numeric.between(1, 5) | (numeric % 1 != 0))
    if invalid.any():
        raise ValueError(f"{field} contains scores outside the integer range 1-5")
    return numeric


def aggregate_human_ratings(human: pd.DataFrame) -> pd.DataFrame:
    """Create the only human-rating table intended for public artifacts."""

    required = {"participant_id", "imagem_id"}
    missing = required - set(human)
    if missing:
        raise ValueError(f"Human ratings are missing columns: {sorted(missing)}")
    score_columns = available_human_score_columns(human)
    if human.duplicated(["participant_id", "imagem_id"]).any():
        raise ValueError("Each participant may rate each image only once")
    data = human.copy()
    for column in score_columns.values():
        data[column] = _validated_ordinal_scores(data[column], column)
    rows: list[dict[str, Any]] = []
    for image_id, group in data.groupby("imagem_id", sort=True):
        row: dict[str, Any] = {"imagem_id": image_id}
        for dimension, column in score_columns.items():
            values = group[column].dropna()
            row[f"human_{dimension}_n"] = len(values)
            row[f"human_{dimension}_median"] = values.median()
            row[f"human_{dimension}_q1"] = values.quantile(0.25)
            row[f"human_{dimension}_q3"] = values.quantile(0.75)
        rows.append(row)
    return pd.DataFrame(rows)


def human_paired_prompt_report(
    human_aggregate: pd.DataFrame,
    image_map: pd.DataFrame,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Compare human medians for neutral-inclusive pairs within each corpus."""

    required_map = {
        "imagem_id",
        "human_corpus_id",
        "human_pair_id",
        "human_prompt_group",
    }
    if missing := required_map - set(image_map):
        raise ValueError(f"Human image map is missing columns: {sorted(missing)}")
    if image_map["imagem_id"].duplicated().any():
        raise ValueError("Human image map contains duplicate imagem_id values")
    median_columns = {
        dimension: f"human_{dimension}_median"
        for dimension in HUMAN_SCORE_COLUMNS
        if f"human_{dimension}_median" in human_aggregate
    }
    if not set(PRIMARY_HUMAN_SCORE_COLUMNS).issubset(median_columns):
        raise ValueError("Human aggregate is missing primary median columns")
    data = human_aggregate.merge(
        image_map[list(required_map)],
        on="imagem_id",
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for corpus_id, corpus in data.groupby("human_corpus_id", sort=True):
        corpus_rows = []
        for dimension, column in median_columns.items():
            pivot = corpus.pivot(
                index="human_pair_id",
                columns="human_prompt_group",
                values=column,
            )
            if not {"inclusive", "neutral"}.issubset(pivot.columns):
                raise ValueError(f"Corpus {corpus_id} has incomplete prompt groups")
            pairs = pivot.dropna(subset=["inclusive", "neutral"])
            differences = pairs["inclusive"] - pairs["neutral"]
            if differences.empty:
                raise ValueError(f"Corpus {corpus_id} has no complete human pairs")
            if np.allclose(differences, 0):
                statistic, p_value, rank_biserial = 0.0, 1.0, 0.0
            else:
                statistic, p_value = wilcoxon(differences, alternative="two-sided")
                nonzero = differences[differences != 0]
                ranks = rankdata(nonzero.abs())
                positive = float(ranks[nonzero.to_numpy() > 0].sum())
                negative = float(ranks[nonzero.to_numpy() < 0].sum())
                rank_biserial = (positive - negative) / (positive + negative)
            ci_low, ci_high = _paired_median_bootstrap(
                differences,
                iterations=bootstrap_iterations,
                seed=seed + len(rows) + len(corpus_rows),
            )
            corpus_rows.append(
                {
                    "human_corpus_id": corpus_id,
                    "dimension": dimension,
                    "analysis_role": (
                        "primary"
                        if dimension in PRIMARY_HUMAN_SCORE_COLUMNS
                        else "exploratory"
                    ),
                    "n_pairs": len(pairs),
                    "median_inclusive": pairs["inclusive"].median(),
                    "median_neutral": pairs["neutral"].median(),
                    "median_paired_difference": differences.median(),
                    "median_difference_ci_2_5": ci_low,
                    "median_difference_ci_97_5": ci_high,
                    "n_inclusive_higher": int((differences > 0).sum()),
                    "n_neutral_higher": int((differences < 0).sum()),
                    "n_ties": int((differences == 0).sum()),
                    "w_statistic": float(statistic),
                    "p_value": float(p_value),
                    "matched_rank_biserial": float(rank_biserial),
                }
            )
        corpus_report = pd.DataFrame(corpus_rows)
        for role, indices in corpus_report.groupby("analysis_role").groups.items():
            corpus_report.loc[indices, "p_value_holm"] = multipletests(
                corpus_report.loc[indices, "p_value"], method="holm"
            )[1]
            corpus_report.loc[indices, "holm_family"] = f"{corpus_id}:{role}"
        rows.extend(corpus_report.to_dict(orient="records"))
    return pd.DataFrame(rows)


def human_participant_prompt_sensitivity(
    ratings: pd.DataFrame,
    image_map: pd.DataFrame,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Summarize within-participant prompt differences without publishing IDs.

    Each participant first receives one mean inclusive-minus-neutral difference
    across the complete image pairs in a corpus. Inference is then performed on
    those participant summaries. This estimates participant response for the
    fixed image sample; it is not a substitute for image-level inference.
    """

    required_ratings = {"participant_id", "imagem_id"}
    if missing := required_ratings - set(ratings):
        raise ValueError(f"Human ratings are missing columns: {sorted(missing)}")
    required_map = {
        "imagem_id",
        "human_corpus_id",
        "human_pair_id",
        "human_prompt_group",
    }
    if missing := required_map - set(image_map):
        raise ValueError(f"Human image map is missing columns: {sorted(missing)}")
    if image_map["imagem_id"].duplicated().any():
        raise ValueError("Human image map contains duplicate imagem_id values")

    score_columns = available_human_score_columns(ratings)
    rating_columns = [
        "participant_id",
        "imagem_id",
        *score_columns.values(),
    ]
    data = ratings[rating_columns].merge(
        image_map[list(required_map)], on="imagem_id", validate="many_to_one"
    )
    rows: list[dict[str, Any]] = []
    for corpus_id, corpus in data.groupby("human_corpus_id", sort=True):
        corpus_rows = []
        for dimension, score_column in score_columns.items():
            pivot = corpus.pivot(
                index=["participant_id", "human_pair_id"],
                columns="human_prompt_group",
                values=score_column,
            )
            pairs = pivot.dropna(subset=["inclusive", "neutral"])
            if pairs.empty:
                raise ValueError(f"Corpus {corpus_id} has no complete human pairs")
            pair_differences = pairs["inclusive"] - pairs["neutral"]
            participant_differences = pair_differences.groupby(
                level="participant_id"
            ).mean()
            pairs_per_participant = pair_differences.groupby(
                level="participant_id"
            ).size()
            if pairs_per_participant.nunique() != 1:
                raise ValueError(
                    f"Corpus {corpus_id} has unequal pair counts per participant"
                )
            if np.allclose(participant_differences, 0):
                statistic, p_value, rank_biserial = 0.0, 1.0, 0.0
            else:
                statistic, p_value = wilcoxon(
                    participant_differences, alternative="two-sided"
                )
                nonzero = participant_differences[participant_differences != 0]
                ranks = rankdata(nonzero.abs())
                positive = float(ranks[nonzero.to_numpy() > 0].sum())
                negative = float(ranks[nonzero.to_numpy() < 0].sum())
                rank_biserial = (positive - negative) / (positive + negative)
            ci_low, ci_high = _paired_median_bootstrap(
                participant_differences,
                iterations=bootstrap_iterations,
                seed=seed + len(rows) + len(corpus_rows),
            )
            corpus_rows.append(
                {
                    "human_corpus_id": corpus_id,
                    "dimension": dimension,
                    "analysis_role": (
                        "primary"
                        if dimension in PRIMARY_HUMAN_SCORE_COLUMNS
                        else "exploratory"
                    ),
                    "n_participants": len(participant_differences),
                    "n_pairs_per_participant": int(pairs_per_participant.iloc[0]),
                    "n_nonzero_participants": int(
                        (participant_differences != 0).sum()
                    ),
                    "median_participant_mean_difference": (
                        participant_differences.median()
                    ),
                    "mean_participant_mean_difference": (
                        participant_differences.mean()
                    ),
                    "median_difference_ci_2_5": ci_low,
                    "median_difference_ci_97_5": ci_high,
                    "n_inclusive_higher": int(
                        (participant_differences > 0).sum()
                    ),
                    "n_neutral_higher": int((participant_differences < 0).sum()),
                    "n_ties": int((participant_differences == 0).sum()),
                    "w_statistic": float(statistic),
                    "p_value": float(p_value),
                    "matched_rank_biserial": float(rank_biserial),
                    "estimand": "participant mean across fixed image pairs",
                }
            )
        corpus_report = pd.DataFrame(corpus_rows)
        for role, indices in corpus_report.groupby("analysis_role").groups.items():
            corpus_report.loc[indices, "p_value_holm"] = multipletests(
                corpus_report.loc[indices, "p_value"], method="holm"
            )[1]
            corpus_report.loc[indices, "holm_family"] = (
                f"{corpus_id}:participant_sensitivity:{role}"
            )
        rows.extend(corpus_report.to_dict(orient="records"))
    return pd.DataFrame(rows)


def ordinal_krippendorff_report(
    ratings: pd.DataFrame,
    unit_col: str = "imagem_id",
    rater_col: str = "participant_id",
    score_columns: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Measure ordinal agreement for any number of raters with missing values."""

    mappings = score_columns or available_human_score_columns(ratings)
    required = {unit_col, rater_col, *mappings.values()}
    missing = required - set(ratings)
    if missing:
        raise ValueError(f"Ratings are missing columns: {sorted(missing)}")
    if ratings.duplicated([rater_col, unit_col]).any():
        raise ValueError("Duplicate rater/unit observations are not allowed")
    rows = []
    for dimension, column in mappings.items():
        data = ratings[[unit_col, rater_col, column]].copy()
        data[column] = _validated_ordinal_scores(data[column], column)
        pivot = data.pivot(index=unit_col, columns=rater_col, values=column)
        matrix = pivot.to_numpy(dtype=float).T
        valid_units = int(((~np.isnan(matrix)).sum(axis=0) >= 2).sum())
        try:
            alpha = float(
                krippendorff.alpha(
                    reliability_data=matrix,
                    level_of_measurement="ordinal",
                )
            )
        except (ValueError, ZeroDivisionError):
            alpha = np.nan
        rows.append(
            {
                "dimension": dimension,
                "n_raters": pivot.shape[1],
                "n_units": pivot.shape[0],
                "n_units_with_overlap": valid_units,
                "krippendorff_alpha_ordinal": alpha,
            }
        )
    return pd.DataFrame(rows)


def judge_repeatability_report(pointwise: pd.DataFrame) -> pd.DataFrame:
    """Report within-judge ordinal stability over repeated image evaluations."""

    required = {
        "provider",
        "model_id",
        "imagem_id",
        "repeat_id",
        "eligible",
        "racial_score",
        "gender_score",
    }
    missing = required - set(pointwise)
    if missing:
        raise ValueError(f"Judge table is missing columns: {sorted(missing)}")
    data = pointwise.loc[pointwise["eligible"].astype(bool)].copy()
    data["judge_id"] = data["provider"].astype(str) + ":" + data["model_id"].astype(str)
    rows = []
    for judge_id, group in data.groupby("judge_id"):
        report = ordinal_krippendorff_report(
            group,
            unit_col="imagem_id",
            rater_col="repeat_id",
            score_columns={"racial": "racial_score", "gender": "gender_score"},
        )
        report.insert(0, "judge_id", judge_id)
        rows.extend(report.to_dict(orient="records"))
    return pd.DataFrame(rows)


def judge_interrater_report(pointwise: pd.DataFrame) -> pd.DataFrame:
    """Report ordinal agreement across judges after aggregating repetitions."""

    required = {
        "provider",
        "model_id",
        "imagem_id",
        "eligible",
        "racial_score",
        "gender_score",
    }
    missing = required - set(pointwise)
    if missing:
        raise ValueError(f"Judge table is missing columns: {sorted(missing)}")
    data = pointwise.loc[pointwise["eligible"].astype(bool)].copy()
    data["judge_id"] = (
        data["provider"].astype(str) + ":" + data["model_id"].astype(str)
    )
    aggregated = (
        data.groupby(["judge_id", "imagem_id"], as_index=False)[
            ["racial_score", "gender_score"]
        ]
        .median()
    )
    return ordinal_krippendorff_report(
        aggregated,
        unit_col="imagem_id",
        rater_col="judge_id",
        score_columns={"racial": "racial_score", "gender": "gender_score"},
    )


def judge_clip_convergence(
    pointwise: pd.DataFrame,
    images: pd.DataFrame,
    bootstrap_iterations: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare each judge with CLIP margins without treating CLIP as truth."""

    required_judge = {
        "provider",
        "model_id",
        "imagem_id",
        "eligible",
        "racial_score",
        "gender_score",
    }
    required_images = {
        "imagem_id",
        "clip_racial_diversity_margin",
        "clip_gender_diversity_margin",
    }
    if missing := required_judge - set(pointwise):
        raise ValueError(f"Judge table is missing columns: {sorted(missing)}")
    if missing := required_images - set(images):
        raise ValueError(f"Image table is missing columns: {sorted(missing)}")
    judge = pointwise.loc[pointwise["eligible"].astype(bool)].copy()
    judge["judge_id"] = (
        judge["provider"].astype(str) + ":" + judge["model_id"].astype(str)
    )
    judge = (
        judge.groupby(["judge_id", "imagem_id"], as_index=False)[
            ["racial_score", "gender_score"]
        ]
        .median()
    )
    merged = judge.merge(
        images[list(required_images)], on="imagem_id", validate="many_to_one"
    )
    rows = []
    for judge_id, group in merged.groupby("judge_id"):
        for dimension in ("racial", "gender"):
            margin = f"clip_{dimension}_diversity_margin"
            valid = group[[f"{dimension}_score", margin]].dropna()
            rho, p_value, ci_low, ci_high = _spearman_bootstrap(
                valid[f"{dimension}_score"],
                valid[margin],
                iterations=bootstrap_iterations,
                seed=seed,
            )
            rows.append(
                {
                    "judge_id": judge_id,
                    "dimension": dimension,
                    "n": len(valid),
                    "spearman_rho": rho,
                    "spearman_p_value": p_value,
                    "spearman_ci_2_5": ci_low,
                    "spearman_ci_97_5": ci_high,
                    "reference_role": "CLIP semantic margin; not Ground Truth",
                }
            )
    return pd.DataFrame(rows)


def _spearman_bootstrap(
    x: pd.Series,
    y: pd.Series,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[float, float, float, float]:
    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan
    rho, p_value = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        indices = rng.integers(0, len(x), len(x))
        sampled_x = x.iloc[indices]
        sampled_y = y.iloc[indices]
        if sampled_x.nunique() < 2 or sampled_y.nunique() < 2:
            continue
        estimate, _ = spearmanr(sampled_x, sampled_y)
        if np.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return float(rho), float(p_value), np.nan, np.nan
    return (
        float(rho),
        float(p_value),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def _partial_spearman_bootstrap(
    x: pd.Series,
    y: pd.Series,
    covariate: pd.Series,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[float, float, float, float]:
    """Estimate rank correlation after residualizing one numeric covariate."""

    data = pd.DataFrame({"x": x, "y": y, "z": covariate}).apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    if len(data) < 4 or data["z"].nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan

    def estimate(sample: pd.DataFrame) -> tuple[float, float]:
        ranked = sample.rank(method="average")
        design = np.column_stack(
            [np.ones(len(ranked)), ranked["z"].to_numpy(dtype=float)]
        )
        residual_x = ranked["x"] - design @ np.linalg.lstsq(
            design, ranked["x"], rcond=None
        )[0]
        residual_y = ranked["y"] - design @ np.linalg.lstsq(
            design, ranked["y"], rcond=None
        )[0]
        if residual_x.nunique() < 2 or residual_y.nunique() < 2:
            return np.nan, np.nan
        result = pearsonr(residual_x, residual_y)
        return float(result.statistic), float(result.pvalue)

    rho, p_value = estimate(data)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        sampled = data.iloc[rng.integers(0, len(data), len(data))]
        if min(sampled[column].nunique() for column in ("x", "y", "z")) < 2:
            continue
        sampled_rho, _ = estimate(sampled)
        if np.isfinite(sampled_rho):
            estimates.append(sampled_rho)
    if not estimates:
        return rho, p_value, np.nan, np.nan
    return (
        rho,
        p_value,
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def _holm_with_missing(values: pd.Series) -> pd.Series:
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    valid = pd.to_numeric(values, errors="coerce").notna()
    if valid.any():
        adjusted.loc[valid] = multipletests(
            pd.to_numeric(values.loc[valid]), method="holm"
        )[1]
    return adjusted


def human_framework_convergence(
    human_aggregate: pd.DataFrame,
    images: pd.DataFrame,
    simpson_by_image: pd.DataFrame,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Compare human medians with matching CLIP and facial-composition metrics.

    Results are separated by corpus. CLIP is the primary comparison family;
    DeepFace Simpson indices are secondary and include a face-count sensitivity
    analysis because the index is structurally dependent on detected faces.
    """

    required_human = {
        "imagem_id",
        "human_racial_median",
        "human_gender_median",
    }
    required_images = {
        "imagem_id",
        "dataset_id",
        "num_faces",
        "clip_racial_diversity_margin",
        "clip_gender_diversity_margin",
    }
    required_simpson = {
        "imagem_id",
        "deepface_racial_simpson",
        "deepface_gender_simpson",
    }
    if missing := required_human - set(human_aggregate):
        raise ValueError(f"Human aggregate is missing columns: {sorted(missing)}")
    if missing := required_images - set(images):
        raise ValueError(f"Image table is missing columns: {sorted(missing)}")
    if missing := required_simpson - set(simpson_by_image):
        raise ValueError(f"Simpson table is missing columns: {sorted(missing)}")
    for name, table in (
        ("human aggregate", human_aggregate),
        ("image table", images),
        ("Simpson table", simpson_by_image),
    ):
        if table["imagem_id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate imagem_id values")

    merged = (
        human_aggregate[list(required_human)]
        .merge(images[list(required_images)], on="imagem_id", validate="one_to_one")
        .merge(
            simpson_by_image[list(required_simpson)],
            on="imagem_id",
            how="left",
            validate="one_to_one",
        )
    )
    metrics = (
        (
            "racial",
            "clip_racial_diversity_margin",
            "clip_primary",
            "CLIP semantic margin; not Ground Truth",
        ),
        (
            "gender",
            "clip_gender_diversity_margin",
            "clip_primary",
            "CLIP semantic margin; not Ground Truth",
        ),
        (
            "racial",
            "deepface_racial_simpson",
            "deepface_simpson_secondary",
            "DeepFace facial-composition index; not demographic truth",
        ),
        (
            "gender",
            "deepface_gender_simpson",
            "deepface_simpson_secondary",
            "DeepFace facial-composition index; not demographic truth",
        ),
    )
    rows: list[dict[str, Any]] = []
    for dataset_id, group in merged.groupby("dataset_id", sort=True):
        for metric_index, (dimension, metric, family, role) in enumerate(metrics):
            human_column = f"human_{dimension}_median"
            valid = group[[human_column, metric, "num_faces"]].apply(
                pd.to_numeric, errors="coerce"
            ).dropna(subset=[human_column, metric])
            rho, p_value, ci_low, ci_high = _spearman_bootstrap(
                valid[human_column],
                valid[metric],
                iterations=bootstrap_iterations,
                seed=seed + metric_index,
            )
            partial = (np.nan, np.nan, np.nan, np.nan)
            if family == "deepface_simpson_secondary":
                partial = _partial_spearman_bootstrap(
                    valid[human_column],
                    valid[metric],
                    valid["num_faces"],
                    iterations=bootstrap_iterations,
                    seed=seed + metric_index,
                )
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "dimension": dimension,
                    "automated_metric": metric,
                    "analysis_family": family,
                    "n": len(valid),
                    "spearman_rho": rho,
                    "spearman_p_value": p_value,
                    "spearman_ci_2_5": ci_low,
                    "spearman_ci_97_5": ci_high,
                    "partial_spearman_rho_face_count": partial[0],
                    "partial_spearman_p_value_face_count": partial[1],
                    "partial_spearman_ci_2_5_face_count": partial[2],
                    "partial_spearman_ci_97_5_face_count": partial[3],
                    "reference_role": role,
                }
            )
    report = pd.DataFrame(rows)
    if report.empty:
        return report
    report["holm_family"] = (
        report["dataset_id"].astype(str)
        + ":"
        + report["analysis_family"].astype(str)
    )
    report["spearman_p_value_holm"] = report.groupby(
        "holm_family", group_keys=False
    )["spearman_p_value"].apply(_holm_with_missing)
    report["partial_holm_family"] = np.where(
        report["analysis_family"].eq("deepface_simpson_secondary"),
        report["dataset_id"].astype(str) + ":deepface_simpson_partial",
        pd.NA,
    )
    report["partial_spearman_p_value_face_count_holm"] = np.nan
    partial_rows = report["partial_holm_family"].notna()
    if partial_rows.any():
        report.loc[
            partial_rows, "partial_spearman_p_value_face_count_holm"
        ] = report.loc[partial_rows].groupby(
            "partial_holm_family", group_keys=False
        )["partial_spearman_p_value_face_count"].apply(_holm_with_missing)
    return report


def paired_simpson_prompt_report(
    images: pd.DataFrame,
    simpson_by_image: pd.DataFrame,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Compare classifier-based Simpson indices within neutral-inclusive pairs."""

    required_images = {"imagem_id", "dataset_id", "pair_id", "tipo_prompt"}
    required_simpson = {
        "imagem_id",
        "deepface_racial_simpson",
        "deepface_gender_simpson",
    }
    if missing := required_images - set(images):
        raise ValueError(f"Image table is missing columns: {sorted(missing)}")
    if missing := required_simpson - set(simpson_by_image):
        raise ValueError(f"Simpson table is missing columns: {sorted(missing)}")
    require_unique_images(images)
    if simpson_by_image["imagem_id"].duplicated().any():
        raise ValueError("Simpson table contains duplicate imagem_id values")

    merged = images[list(required_images)].merge(
        simpson_by_image[list(required_simpson)],
        on="imagem_id",
        how="left",
        validate="one_to_one",
    )
    merged["grupo"] = merged["tipo_prompt"].map(normalize_prompt_group)
    metrics = (
        ("racial", "deepface_racial_simpson"),
        ("gender", "deepface_gender_simpson"),
    )
    rows: list[dict[str, Any]] = []
    for dataset_id, corpus in merged.groupby("dataset_id", sort=True):
        corpus_rows: list[dict[str, Any]] = []
        for metric_index, (dimension, metric) in enumerate(metrics):
            pivot = corpus.pivot_table(
                index="pair_id", columns="grupo", values=metric, aggfunc="first"
            )
            pairs = pivot.dropna(subset=["inclusive", "neutral"])
            if pairs.empty:
                raise ValueError(
                    f"Corpus {dataset_id} has no complete Simpson pairs for {metric}"
                )
            differences = pairs["inclusive"] - pairs["neutral"]
            if np.allclose(differences, 0):
                statistic, p_value, rank_biserial = 0.0, 1.0, 0.0
            else:
                statistic, p_value = wilcoxon(differences, alternative="two-sided")
                nonzero = differences[differences != 0]
                ranks = rankdata(nonzero.abs())
                positive = float(ranks[nonzero.to_numpy() > 0].sum())
                negative = float(ranks[nonzero.to_numpy() < 0].sum())
                rank_biserial = (positive - negative) / (positive + negative)
            ci_low, ci_high = _paired_median_bootstrap(
                differences,
                iterations=bootstrap_iterations,
                seed=seed + metric_index,
            )
            corpus_rows.append(
                {
                    "dataset_id": dataset_id,
                    "dimension": dimension,
                    "automated_metric": metric,
                    "n_pairs": len(pairs),
                    "median_inclusive": pairs["inclusive"].median(),
                    "median_neutral": pairs["neutral"].median(),
                    "median_paired_difference": differences.median(),
                    "median_difference_ci_2_5": ci_low,
                    "median_difference_ci_97_5": ci_high,
                    "n_inclusive_higher": int((differences > 0).sum()),
                    "n_neutral_higher": int((differences < 0).sum()),
                    "n_ties": int((differences == 0).sum()),
                    "w_statistic": float(statistic),
                    "p_value": float(p_value),
                    "matched_rank_biserial": float(rank_biserial),
                    "difference_convention": "inclusive-minus-neutral",
                }
            )
        corpus_report = pd.DataFrame(corpus_rows)
        corpus_report["p_value_holm"] = multipletests(
            corpus_report["p_value"], method="holm"
        )[1]
        corpus_report["holm_family"] = f"{dataset_id}:simpson_secondary"
        rows.extend(corpus_report.to_dict(orient="records"))
    return pd.DataFrame(rows)


def human_judge_convergence(
    human_aggregate: pd.DataFrame,
    pointwise: pd.DataFrame,
    bootstrap_iterations: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare per-judge median scores with the human perceptual reference."""

    required_human = {
        "imagem_id",
        "human_racial_median",
        "human_gender_median",
    }
    required_judge = {
        "provider",
        "model_id",
        "imagem_id",
        "eligible",
        "racial_score",
        "gender_score",
    }
    if missing := required_human - set(human_aggregate):
        raise ValueError(f"Human aggregate is missing columns: {sorted(missing)}")
    if missing := required_judge - set(pointwise):
        raise ValueError(f"Judge table is missing columns: {sorted(missing)}")
    judge = pointwise.loc[pointwise["eligible"].astype(bool)].copy()
    judge["judge_id"] = judge["provider"].astype(str) + ":" + judge["model_id"].astype(str)
    judge = (
        judge.groupby(["judge_id", "imagem_id"], as_index=False)[
            ["racial_score", "gender_score"]
        ]
        .median()
    )
    merged = judge.merge(human_aggregate, on="imagem_id", validate="many_to_one")
    rows = []
    for judge_id, group in merged.groupby("judge_id"):
        for dimension in ("racial", "gender"):
            valid = group[
                [f"human_{dimension}_median", f"{dimension}_score"]
            ].dropna()
            rho, p_value, ci_low, ci_high = _spearman_bootstrap(
                valid[f"human_{dimension}_median"],
                valid[f"{dimension}_score"],
                iterations=bootstrap_iterations,
                seed=seed,
            )
            error = (
                valid[f"human_{dimension}_median"] - valid[f"{dimension}_score"]
            ).abs()
            rows.append(
                {
                    "judge_id": judge_id,
                    "dimension": dimension,
                    "n": len(valid),
                    "spearman_rho": rho,
                    "spearman_p_value": p_value,
                    "spearman_ci_2_5": ci_low,
                    "spearman_ci_97_5": ci_high,
                    "mae": error.mean() if len(error) else np.nan,
                    "within_one_rate": (error <= 1).mean() if len(error) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def pairwise_judge_report(
    pairwise: pd.DataFrame,
    images: pd.DataFrame,
) -> pd.DataFrame:
    """Measure position consistency and prompt preference without duplicated pairs."""

    required = {
        "provider",
        "model_id",
        "dataset_id",
        "pair_id",
        "order_id",
        "racial_winner_id",
        "gender_winner_id",
    }
    missing = required - set(pairwise)
    if missing:
        raise ValueError(f"Pairwise table is missing columns: {sorted(missing)}")
    group_map = images.set_index("imagem_id")["tipo_prompt"].map(normalize_prompt_group)
    data = pairwise.copy()
    data["judge_id"] = data["provider"].astype(str) + ":" + data["model_id"].astype(str)
    rows = []
    keys = ["judge_id", "dataset_id", "pair_id"]
    for judge_id, judge_data in data.groupby("judge_id"):
        for dimension in ("racial", "gender"):
            winners = f"{dimension}_winner_id"
            pair_rows = []
            for _, group in judge_data.groupby(keys[1:]):
                values = group[winners].tolist()
                consistent = len(values) == 2 and values[0] == values[1]
                winner = values[0] if consistent else None
                prompt_group = group_map.get(winner) if winner not in {None, "tie", "not_evaluable"} else winner
                pair_rows.append((consistent, prompt_group))
            consistent_count = sum(item[0] for item in pair_rows)
            inclusive = sum(item == (True, "inclusive") for item in pair_rows)
            neutral = sum(item == (True, "neutral") for item in pair_rows)
            decisive = inclusive + neutral
            p_value = (
                float(binomtest(inclusive, decisive, p=0.5).pvalue)
                if decisive
                else np.nan
            )
            rows.append(
                {
                    "judge_id": judge_id,
                    "dimension": dimension,
                    "n_pairs": len(pair_rows),
                    "position_consistency_rate": (
                        consistent_count / len(pair_rows) if pair_rows else np.nan
                    ),
                    "inclusive_wins": inclusive,
                    "neutral_wins": neutral,
                    "decisive_pairs": decisive,
                    "binomial_p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


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


def per_model_wilcoxon(
    images: pd.DataFrame,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Run descriptive paired tests for each generator and one Holm family."""

    rows = []
    for model_index, (model, subset) in enumerate(
        images.dropna(subset=["modelo_ia"]).groupby("modelo_ia", sort=True)
    ):
        try:
            report = wilcoxon_reports(
                subset,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed + model_index * len(PRIMARY_METRICS),
            ).drop(columns=["p_value_holm"])
        except ValueError:
            continue
        report.insert(0, "modelo_ia", model)
        rows.extend(report.to_dict(orient="records"))
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_value_holm_family"] = multipletests(
            result["p_value"], method="holm"
        )[1]
        result["analysis_role"] = "descriptive_small_n"
    return result


def normalized_simpson(values: pd.Series, categories: int) -> float:
    """Return the legacy normalized plug-in Simpson index.

    This function remains available for backward compatibility. Canonical
    reports use :func:`unbiased_gini_simpson` because the plug-in estimate is
    structurally compressed when only a few faces are detected.
    """

    valid = values.dropna()
    if len(valid) < 2:
        return float("nan")
    if categories < 2:
        raise ValueError("categories must be at least two")
    proportions = valid.value_counts(normalize=True)
    raw = 1.0 - float(np.square(proportions).sum())
    return raw / (1.0 - 1.0 / categories)


def unbiased_gini_simpson(values: pd.Series) -> float:
    """Return the unbiased Gini-Simpson estimate, or NaN for fewer than two."""

    valid = values.dropna()
    n = len(valid)
    if n < 2:
        return float("nan")
    counts = valid.value_counts().to_numpy(dtype=float)
    same_category_pairs = float(np.sum(counts * (counts - 1.0)))
    return 1.0 - same_category_pairs / (n * (n - 1.0))


def face_diversity_by_image(faces: pd.DataFrame) -> pd.DataFrame:
    """Calculate unbiased classifier-based facial-composition indices."""

    rows = []
    for image_id, subset in faces.groupby("imagem_id"):
        rows.append(
            {
                "imagem_id": image_id,
                "deepface_racial_simpson": unbiased_gini_simpson(subset["raca"]),
                "deepface_gender_simpson": unbiased_gini_simpson(
                    subset["genero"]
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
