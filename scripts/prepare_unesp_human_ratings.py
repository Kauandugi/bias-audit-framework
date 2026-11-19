"""Convert the private UNESP Google Forms export into publishable aggregates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from biasauditfw.human_forms import (
    filter_consented_responses,
    google_forms_responses_to_long,
)
from biasauditfw.statistics import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_RANDOM_SEED,
    aggregate_human_ratings,
    human_framework_convergence,
    human_judge_convergence,
    human_paired_prompt_report,
    human_participant_prompt_sensitivity,
    ordinal_krippendorff_report,
)


def build_validation_metadata(
    ratings: pd.DataFrame,
    consent_summary: dict[str, int | str],
    *,
    form_code: str,
    authorized_for_aggregate_use: bool,
    authorization_basis: str | None,
    ethics_opinion_number: str | None,
    ethics_opinion_date: str | None,
) -> dict[str, object]:
    """Describe admissibility without exposing participant-level attributes."""

    if authorized_for_aggregate_use and not authorization_basis:
        raise ValueError(
            "--authorization-basis is required when aggregate use is authorized"
        )
    return {
        "form_code": form_code,
        "participants": int(ratings["participant_id"].nunique()),
        "images": int(ratings["imagem_id"].nunique()),
        "private_rows": len(ratings),
        "profile_fields_exported": False,
        "individual_data_published": False,
        "publication_scope": "aggregate_only",
        **{
            key: value
            for key, value in consent_summary.items()
            if key != "consent_column"
        },
        "ethics_opinion_number": ethics_opinion_number,
        "ethics_opinion_date": ethics_opinion_date,
        "authorization_status": (
            "authorized_for_aggregate_use"
            if authorized_for_aggregate_use
            else "not_authorized_for_final_use"
        ),
        "authorization_basis": authorization_basis,
        "usable_as_final_evidence": authorized_for_aggregate_use,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    responses_group = parser.add_mutually_exclusive_group(required=True)
    responses_group.add_argument("--responses-csv", type=Path)
    responses_group.add_argument("--responses-xlsx", type=Path)
    parser.add_argument("--responses-sheet", default="Form Responses 1")
    parser.add_argument("--image-map", type=Path, required=True)
    parser.add_argument("--private-long-csv", type=Path, required=True)
    parser.add_argument("--public-output-dir", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=48)
    parser.add_argument("--form-code", default="UNESP-2026")
    parser.add_argument("--judge-pointwise", type=Path)
    parser.add_argument("--audit-images", type=Path, nargs="+")
    parser.add_argument("--simpson-by-image", type=Path, nargs="+")
    parser.add_argument("--authorized-for-aggregate-use", action="store_true")
    parser.add_argument("--authorization-basis")
    parser.add_argument("--ethics-opinion-number")
    parser.add_argument("--ethics-opinion-date")
    args = parser.parse_args()

    if args.responses_xlsx:
        responses = pd.read_excel(
            args.responses_xlsx,
            sheet_name=args.responses_sheet,
            engine="openpyxl",
        )
    else:
        responses = pd.read_csv(args.responses_csv)
    responses, consent_summary = filter_consented_responses(responses)
    image_map = pd.read_csv(args.image_map)
    ratings = google_forms_responses_to_long(
        responses,
        image_map,
        expected_images=args.expected_images,
        form_code=args.form_code,
        require_consent=False,
    )
    aggregate = aggregate_human_ratings(ratings)
    agreement = ordinal_krippendorff_report(ratings)

    args.private_long_csv.parent.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(args.private_long_csv, index=False)
    args.public_output_dir.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(
        args.public_output_dir / "human_ratings_aggregate.csv", index=False
    )
    agreement.to_csv(
        args.public_output_dir / "human_interrater_agreement.csv", index=False
    )
    human_paired_prompt_report(
        aggregate,
        image_map,
        bootstrap_iterations=DEFAULT_BOOTSTRAP_ITERATIONS,
        seed=DEFAULT_RANDOM_SEED,
    ).to_csv(
        args.public_output_dir / "human_paired_prompt_effects.csv", index=False
    )
    human_participant_prompt_sensitivity(
        ratings,
        image_map,
        bootstrap_iterations=DEFAULT_BOOTSTRAP_ITERATIONS,
        seed=DEFAULT_RANDOM_SEED,
    ).to_csv(
        args.public_output_dir / "human_participant_sensitivity.csv", index=False
    )
    if args.judge_pointwise:
        judges = pd.read_csv(args.judge_pointwise)
        human_judge_convergence(aggregate, judges).to_csv(
            args.public_output_dir / "human_judge_convergence.csv", index=False
        )
    if bool(args.audit_images) != bool(args.simpson_by_image):
        raise ValueError(
            "--audit-images and --simpson-by-image must be provided together"
        )
    if args.audit_images:
        images = pd.concat(
            [pd.read_csv(path) for path in args.audit_images], ignore_index=True
        )
        simpson = pd.concat(
            [pd.read_csv(path) for path in args.simpson_by_image],
            ignore_index=True,
        )
        human_framework_convergence(
            aggregate,
            images,
            simpson,
            bootstrap_iterations=DEFAULT_BOOTSTRAP_ITERATIONS,
            seed=DEFAULT_RANDOM_SEED,
        ).to_csv(
            args.public_output_dir / "human_framework_convergence.csv",
            index=False,
        )

    summary = build_validation_metadata(
        ratings,
        consent_summary,
        form_code=args.form_code,
        authorized_for_aggregate_use=args.authorized_for_aggregate_use,
        authorization_basis=args.authorization_basis,
        ethics_opinion_number=args.ethics_opinion_number,
        ethics_opinion_date=args.ethics_opinion_date,
    )
    (args.public_output_dir / "human_validation_metadata.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(
        "Private long-form ratings and public aggregate artifacts were written; "
        "profile fields were discarded."
    )


if __name__ == "__main__":
    main()
