"""Regenerate the frozen statistical artifacts used by the final TCC release."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from biasauditfw.statistics import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_RANDOM_SEED,
    aggregate_human_ratings,
    face_diversity_by_image,
    human_framework_convergence,
    human_paired_prompt_report,
    human_participant_prompt_sensitivity,
    mann_whitney_reports,
    ordinal_krippendorff_report,
    paired_simpson_prompt_report,
    per_model_mann_whitney,
    per_model_wilcoxon,
    wilcoxon_reports,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_HUMAN = ROOT / "data" / "human-unesp-aggregate"
PRIVATE_HUMAN = ROOT / "data" / "private" / "unesp-2026"
CORPORA = (
    ROOT / "data" / "schema2",
    ROOT / "data" / "schema2-vitor",
)


def _image_face_summary(images: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_id, corpus in images.groupby("dataset_id", sort=True):
        for prompt_group, group in corpus.groupby("tipo_prompt", sort=True):
            counts = pd.to_numeric(group["num_faces"], errors="raise")
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "prompt_group": prompt_group,
                    "n_images": len(group),
                    "mean_faces": counts.mean(),
                    "median_faces": counts.median(),
                    "zero_face_images": int((counts == 0).sum()),
                    "one_face_images": int((counts == 1).sum()),
                    "multi_face_images": int((counts >= 2).sum()),
                    "total_faces": int(counts.sum()),
                }
            )
    return pd.DataFrame(rows)


def _face_category_composition(
    images: pd.DataFrame, faces: pd.DataFrame
) -> pd.DataFrame:
    image_groups = images[["imagem_id", "dataset_id", "tipo_prompt"]].copy()
    data = faces.merge(image_groups, on=["imagem_id", "dataset_id"], validate="many_to_one")
    rows: list[dict[str, object]] = []
    for (dataset_id, prompt_group), group in data.groupby(
        ["dataset_id", "tipo_prompt"], sort=True
    ):
        for dimension, column in (("racial", "raca"), ("gender", "genero")):
            counts = group[column].dropna().astype(str).value_counts().sort_index()
            total = int(counts.sum())
            for category, count in counts.items():
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "prompt_group": prompt_group,
                        "dimension": dimension,
                        "category": category,
                        "count": int(count),
                        "percentage": 100.0 * float(count) / total,
                        "total_classified_faces": total,
                        "reference_role": "classifier prediction; not demographic truth",
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    images_by_corpus: list[pd.DataFrame] = []
    simpson_by_corpus: list[pd.DataFrame] = []
    faces_by_corpus: list[pd.DataFrame] = []
    for corpus_dir in CORPORA:
        images = pd.read_csv(corpus_dir / "audit_images.csv")
        faces = pd.read_csv(corpus_dir / "audit_faces.csv")
        simpson = face_diversity_by_image(faces)
        simpson.to_csv(corpus_dir / "deepface_simpson_by_image.csv", index=False)
        images_by_corpus.append(images)
        simpson_by_corpus.append(simpson)
        faces_by_corpus.append(faces)

        wilcoxon_reports(
            images,
            bootstrap_iterations=DEFAULT_BOOTSTRAP_ITERATIONS,
            seed=DEFAULT_RANDOM_SEED,
        ).to_csv(corpus_dir / "wilcoxon_primary.csv", index=False)
        mann_whitney_reports(images).to_csv(
            corpus_dir / "mann_whitney_sensitivity.csv", index=False
        )
        per_model_mann_whitney(images).to_csv(
            corpus_dir / "mann_whitney_by_model.csv", index=False
        )
        per_model_wilcoxon(
            images,
            bootstrap_iterations=DEFAULT_BOOTSTRAP_ITERATIONS,
            seed=DEFAULT_RANDOM_SEED,
        ).to_csv(corpus_dir / "wilcoxon_by_model.csv", index=False)

    images = pd.concat(images_by_corpus, ignore_index=True)
    simpson = pd.concat(simpson_by_corpus, ignore_index=True)
    faces = pd.concat(faces_by_corpus, ignore_index=True)
    simpson_primary = paired_simpson_prompt_report(
        images,
        simpson,
        bootstrap_iterations=DEFAULT_BOOTSTRAP_ITERATIONS,
        seed=DEFAULT_RANDOM_SEED,
    )
    simpson_primary["analysis_variant"] = "all_prompts"
    simpson_primary["simpson_estimator"] = "unbiased_gini_simpson"
    simpson_primary.to_csv(
        PUBLIC_HUMAN / "deepface_simpson_paired.csv", index=False
    )
    simpson_without_p8 = paired_simpson_prompt_report(
        images.loc[images["prompt_id"].astype(str).str.upper().ne("P8")],
        simpson,
        bootstrap_iterations=DEFAULT_BOOTSTRAP_ITERATIONS,
        seed=DEFAULT_RANDOM_SEED,
    )
    simpson_without_p8["analysis_variant"] = "exclude_P8"
    simpson_without_p8["simpson_estimator"] = "unbiased_gini_simpson"
    simpson_without_p8.to_csv(
        PUBLIC_HUMAN / "deepface_simpson_sensitivity.csv", index=False
    )
    _image_face_summary(images).to_csv(
        PUBLIC_HUMAN / "image_face_count_summary.csv", index=False
    )
    _face_category_composition(images, faces).to_csv(
        PUBLIC_HUMAN / "face_category_composition.csv", index=False
    )

    ratings = pd.read_csv(PRIVATE_HUMAN / "human_ratings_long.csv")
    image_map = pd.read_csv(PRIVATE_HUMAN / "form_image_map_private.csv")
    aggregate = aggregate_human_ratings(ratings)
    aggregate.to_csv(PUBLIC_HUMAN / "human_ratings_aggregate.csv", index=False)
    ordinal_krippendorff_report(ratings).to_csv(
        PUBLIC_HUMAN / "human_interrater_agreement.csv", index=False
    )
    human_paired_prompt_report(
        aggregate,
        image_map,
        bootstrap_iterations=DEFAULT_BOOTSTRAP_ITERATIONS,
        seed=DEFAULT_RANDOM_SEED,
    ).to_csv(PUBLIC_HUMAN / "human_paired_prompt_effects.csv", index=False)
    human_participant_prompt_sensitivity(
        ratings,
        image_map,
        bootstrap_iterations=DEFAULT_BOOTSTRAP_ITERATIONS,
        seed=DEFAULT_RANDOM_SEED,
    ).to_csv(
        PUBLIC_HUMAN / "human_participant_sensitivity.csv", index=False
    )
    human_framework_convergence(
        aggregate,
        images,
        simpson,
        bootstrap_iterations=DEFAULT_BOOTSTRAP_ITERATIONS,
        seed=DEFAULT_RANDOM_SEED,
    ).to_csv(PUBLIC_HUMAN / "human_framework_convergence.csv", index=False)

    metadata_path = PUBLIC_HUMAN / "human_validation_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "collection_started_on": "2026-10-09",
            "collection_ended_on": "2026-10-15",
            "collection_preceded_final_ethics_opinion": True,
            "ethics_caae": "02077126.7.0000.8030",
            "ethics_opinion_number": "8.854.629",
            "ethics_opinion_date": "2026-11-05",
            "ethics_opinion_status": "approved",
            "authorization_status": "authorized_for_aggregate_use",
            "authorization_basis": (
                "Parecer final favoravel n. 8.854.629, adotado como referencia "
                "etica do projeto e do instrumento conforme orientacao academica, "
                "e autorizacao especifica por escrito, informada pelo pesquisador, "
                "para uso academico agregado das respostas."
            ),
            "authorization_document_public": False,
            "usable_as_final_evidence": True,
            "bootstrap_iterations": DEFAULT_BOOTSTRAP_ITERATIONS,
            "random_seed": DEFAULT_RANDOM_SEED,
            "simpson_estimator": "unbiased_gini_simpson",
            "simpson_minimum_faces": 2,
            "simpson_p8_sensitivity_reported": True,
            "participant_level_sensitivity_published_as_aggregate_only": True,
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print("Final statistical artifacts regenerated with 10,000 bootstrap samples.")


if __name__ == "__main__":
    main()
