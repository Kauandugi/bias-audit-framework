"""Import blinded Google Forms ratings without retaining participant profiles."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any

import pandas as pd


FORM_SCORE_COLUMNS = {
    "racial": "racial_score",
    "gender": "gender_score",
    "cultural": "cultural_score",
    "body": "body_score",
    "quality": "quality_score",
}

DEFAULT_ACCEPTED_CONSENT_VALUES = ("Sim, concordo",)
_CONSENT_COLUMN_MARKERS = (
    "concordo em participar",
    "consentimento",
)
_DECLINED_CONSENT_VALUES = {
    "nao concordo",
    "nao aceito",
    "nao autorizo",
}

_DIMENSION_ALIASES = {
    "racial": ("diversidade racial",),
    "gender": (
        "equilibrio de genero",
        "diversidade de apresentacao de genero",
        "apresentacao de genero",
    ),
    "cultural": (
        "contexto latino americano",
        "contexto universitario latino americano",
    ),
    "body": ("diversidade corporal",),
    "quality": (
        "qualidade da imagem",
        "qualidade visual",
        "qualidade e coerencia visual",
    ),
}


def _normalize_label(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value).casefold())
    text = "".join(character for character in text if not unicodedata.combining(character))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def detect_form_score_columns(columns: list[str]) -> dict[tuple[int, str], str]:
    """Map Google Forms grid columns to image positions and score fields."""

    detected: dict[tuple[int, str], str] = {}
    for original in columns:
        normalized = _normalize_label(original)
        position_match = re.search(r"\bimagem\s*(\d+)\b", normalized)
        if not position_match:
            continue
        position = int(position_match.group(1))
        dimensions = [
            dimension
            for dimension, aliases in _DIMENSION_ALIASES.items()
            if any(alias in normalized for alias in aliases)
        ]
        if not dimensions:
            continue
        if len(dimensions) > 1:
            raise ValueError(f"Ambiguous form score column: {original!r}")
        key = (position, FORM_SCORE_COLUMNS[dimensions[0]])
        if key in detected:
            raise ValueError(
                f"Duplicate form score column for image {position}: {key[1]}"
            )
        detected[key] = original
    return detected


def detect_consent_column(columns: list[str]) -> str:
    """Locate the consent field without depending on the exported language encoding."""

    candidates = [
        column
        for column in columns
        if any(
            marker in _normalize_label(column)
            for marker in _CONSENT_COLUMN_MARKERS
        )
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Expected exactly one consent column, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def filter_consented_responses(
    responses: pd.DataFrame,
    consent_column: str | None = None,
    accepted_values: tuple[str, ...] = DEFAULT_ACCEPTED_CONSENT_VALUES,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Keep only explicit consent and return a non-identifying audit summary."""

    if responses.empty:
        raise ValueError("The Google Forms export contains no responses")
    column = consent_column or detect_consent_column(
        [str(value) for value in responses.columns]
    )
    if column not in responses:
        raise ValueError(f"Consent column not found: {column!r}")
    accepted = {_normalize_label(value) for value in accepted_values}
    normalized = responses[column].map(_normalize_label)
    accepted_mask = normalized.isin(accepted)
    declined_mask = normalized.isin(_DECLINED_CONSENT_VALUES)
    missing_mask = normalized.eq("")
    unknown_mask = ~(accepted_mask | declined_mask | missing_mask)
    if unknown_mask.any():
        unknown = sorted(set(normalized.loc[unknown_mask]))
        raise ValueError(f"Unknown consent responses: {unknown}")
    filtered = responses.loc[accepted_mask].copy().reset_index(drop=True)
    summary: dict[str, int | str] = {
        "consent_column": column,
        "responses_total": int(len(responses)),
        "responses_consented": int(accepted_mask.sum()),
        "responses_declined": int(declined_mask.sum()),
        "responses_missing_consent": int(missing_mask.sum()),
        "responses_excluded": int((~accepted_mask).sum()),
    }
    return filtered, summary


def validate_form_image_map(
    image_map: pd.DataFrame,
    expected_images: int | None = 48,
) -> pd.DataFrame:
    """Validate the private link between form positions and canonical images."""

    required = {"form_position", "imagem_id"}
    missing = required - set(image_map)
    if missing:
        raise ValueError(f"Image map is missing columns: {sorted(missing)}")
    data = image_map.copy()
    positions = pd.to_numeric(data["form_position"], errors="coerce")
    if positions.isna().any() or (positions % 1 != 0).any():
        raise ValueError("form_position must contain integers")
    data["form_position"] = positions.astype(int)
    if data["form_position"].duplicated().any():
        raise ValueError("form_position must be unique")
    if data["imagem_id"].isna().any() or data["imagem_id"].astype(str).str.strip().eq("").any():
        raise ValueError("imagem_id is required for every form position")
    if data["imagem_id"].duplicated().any():
        raise ValueError("imagem_id must be unique in the form map")
    ordered = data.sort_values("form_position").reset_index(drop=True)
    expected_count = expected_images if expected_images is not None else len(ordered)
    if len(ordered) != expected_count:
        raise ValueError(
            f"Expected {expected_count} mapped images, found {len(ordered)}"
        )
    expected_positions = list(range(1, expected_count + 1))
    if ordered["form_position"].tolist() != expected_positions:
        raise ValueError("form_position must be consecutive from 1")
    return ordered


def google_forms_responses_to_long(
    responses: pd.DataFrame,
    image_map: pd.DataFrame,
    expected_images: int | None = 48,
    form_code: str = "UNESP-2026",
    require_consent: bool = True,
    consent_column: str | None = None,
    accepted_consent_values: tuple[str, ...] = DEFAULT_ACCEPTED_CONSENT_VALUES,
) -> pd.DataFrame:
    """Transform one Google Forms export into a private image-level rating table.

    Columns unrelated to the five image scales are ignored. This intentionally
    excludes timestamps, account fields and participant-profile answers.
    """

    if require_consent:
        responses, _ = filter_consented_responses(
            responses,
            consent_column=consent_column,
            accepted_values=accepted_consent_values,
        )
    elif responses.empty:
        raise ValueError("The Google Forms export contains no responses")
    mapping = validate_form_image_map(image_map, expected_images=expected_images)
    detected = detect_form_score_columns([str(column) for column in responses.columns])
    expected_keys = {
        (position, score_column)
        for position in mapping["form_position"]
        for score_column in FORM_SCORE_COLUMNS.values()
    }
    missing = expected_keys - set(detected)
    if missing:
        preview = sorted(missing)[:8]
        raise ValueError(f"Missing Google Forms score columns: {preview}")
    unexpected_positions = {
        position for position, _ in detected if position not in expected_keys_position(mapping)
    }
    if unexpected_positions:
        raise ValueError(
            f"Google Forms contains unmapped image positions: {sorted(unexpected_positions)}"
        )

    metadata_columns = [
        column
        for column in mapping.columns
        if column not in {"form_position", "imagem_id"}
    ]
    rows: list[dict[str, Any]] = []
    for participant_number, (_, response) in enumerate(responses.iterrows(), start=1):
        participant_id = f"P{participant_number:03d}"
        for mapped in mapping.itertuples(index=False):
            position = int(mapped.form_position)
            row: dict[str, Any] = {
                "participant_id": participant_id,
                "form_code": form_code,
                "form_position": position,
                "imagem_id": str(mapped.imagem_id),
            }
            for score_column in FORM_SCORE_COLUMNS.values():
                source_column = detected[(position, score_column)]
                value = pd.to_numeric(pd.Series([response[source_column]]), errors="coerce").iloc[0]
                if pd.isna(value) or value % 1 != 0 or not 1 <= value <= 5:
                    raise ValueError(
                        f"Invalid score at response {participant_number}, "
                        f"image {position}, field {score_column}"
                    )
                row[score_column] = int(value)
            for column in metadata_columns:
                row[column] = getattr(mapped, column)
            rows.append(row)
    return pd.DataFrame(rows)


def expected_keys_position(image_map: pd.DataFrame) -> set[int]:
    """Return mapped positions as integers for parser diagnostics."""

    return set(image_map["form_position"].astype(int))


def form_dimension_labels() -> Mapping[str, str]:
    """Expose the canonical internal field names used by the importer."""

    return FORM_SCORE_COLUMNS.copy()
