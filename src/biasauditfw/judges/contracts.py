"""Strict response contracts for multimodal judges."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


PAIR_CHOICES = frozenset({"A", "B", "tie", "not_evaluable"})


def _json_object(raw: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Judge response must be a JSON object")
    return value


def _score(value: Any, field: str, eligible: bool) -> int | None:
    if value is None and not eligible:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
        raise ValueError(f"{field} must be an integer from 1 to 5")
    return value


@dataclass(frozen=True)
class PointwiseJudgment:
    """Per-image perceived-diversity scores."""

    eligible: bool
    racial_score: int | None
    gender_score: int | None
    abstain_reason: str | None = None

    @classmethod
    def parse(cls, raw: str | dict[str, Any]) -> "PointwiseJudgment":
        data = _json_object(raw)
        eligible = data.get("eligible")
        if not isinstance(eligible, bool):
            raise ValueError("eligible must be boolean")
        reason = data.get("abstain_reason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("abstain_reason must be a string or null")
        if not eligible and not reason:
            raise ValueError("Ineligible judgments require abstain_reason")
        return cls(
            eligible=eligible,
            racial_score=_score(data.get("racial_score"), "racial_score", eligible),
            gender_score=_score(data.get("gender_score"), "gender_score", eligible),
            abstain_reason=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairwiseJudgment:
    """A/B preference for perceived diversity in a matched image pair."""

    racial_choice: str
    gender_choice: str

    @classmethod
    def parse(cls, raw: str | dict[str, Any]) -> "PairwiseJudgment":
        data = _json_object(raw)
        racial = data.get("racial_choice")
        gender = data.get("gender_choice")
        if racial not in PAIR_CHOICES:
            raise ValueError(f"racial_choice must be one of {sorted(PAIR_CHOICES)}")
        if gender not in PAIR_CHOICES:
            raise ValueError(f"gender_choice must be one of {sorted(PAIR_CHOICES)}")
        return cls(racial_choice=racial, gender_choice=gender)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_pair_choice(choice: str, image_a_id: str, image_b_id: str) -> str | None:
    """Convert positional choices to stable image identifiers."""

    if choice == "A":
        return image_a_id
    if choice == "B":
        return image_b_id
    if choice in {"tie", "not_evaluable"}:
        return choice
    raise ValueError(f"Unknown pair choice: {choice}")
