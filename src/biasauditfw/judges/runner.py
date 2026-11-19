"""Resumable execution and export for multimodal judge tasks."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from ..statistics import normalize_prompt_group
from .contracts import PairwiseJudgment, PointwiseJudgment, normalize_pair_choice
from .prompts import PAIRWISE_PROMPT, POINTWISE_PROMPT, PROMPT_VERSION
from .providers import JudgeProvider


@dataclass(frozen=True)
class PairTask:
    dataset_id: str
    pair_id: str
    image_a_id: str
    image_b_id: str
    image_a_path: Path
    image_b_path: Path
    image_a_hash: str
    image_b_hash: str
    order_id: str


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _inference_config(provider: JudgeProvider) -> dict[str, Any]:
    value = getattr(provider, "inference_config", {})
    if callable(value):
        value = value()
    if not isinstance(value, dict):
        raise TypeError("Provider inference_config must be a dictionary")
    return value


class JudgeRunStore:
    """Append-only JSONL store that skips only successfully parsed tasks."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._successful = {
            row["task_key"]
            for row in self.records()
            if row.get("status") == "success" and row.get("task_key")
        }

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {self.path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record must be an object at line {line_number}")
            rows.append(value)
        return rows

    def is_complete(self, task_key: str) -> bool:
        return task_key in self._successful

    def append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        if record.get("status") == "success":
            self._successful.add(record["task_key"])


def _call_with_retry(
    operation: Callable[[], str],
    attempts: int,
    initial_delay: float,
    sleep: Callable[[float], None],
) -> tuple[str | None, str | None, int]:
    if attempts < 1:
        raise ValueError("attempts must be at least one")
    delay = initial_delay
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return operation(), None, attempt
        except Exception as exc:  # provider SDKs expose different error types
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < attempts:
                sleep(delay)
                delay *= 2
    return None, last_error, attempts


def _is_exhausted_quota(error: str | None) -> bool:
    """Identify rate-limit failures that persisted through all retries."""

    if not error:
        return False
    normalized = error.casefold()
    return any(
        marker in normalized
        for marker in (
            "resource_exhausted",
            "rate_limit_exceeded",
            "rate limit reached",
            "quota exceeded",
            "error code: 429",
        )
    )


def _raise_if_quota_exhausted(provider: JudgeProvider, error: str | None) -> None:
    if _is_exhausted_quota(error):
        raise RuntimeError(
            f"Provider {provider.provider_name}/{provider.model_id} exhausted its "
            "quota after all retries; successful checkpoints were preserved."
        )


def _base_record(
    provider: JudgeProvider,
    task_key: str,
    task_type: str,
    repeat_id: int,
    order_id: str | None,
) -> dict[str, Any]:
    inference_config = _inference_config(provider)
    return {
        "task_key": task_key,
        "task_type": task_type,
        "provider": provider.provider_name,
        "model_id": provider.model_id,
        "model_revision": provider.model_revision,
        "prompt_version": PROMPT_VERSION,
        "temperature": 0,
        "seed": getattr(provider, "seed", None),
        "inference_config_json": json.dumps(
            inference_config, sort_keys=True, ensure_ascii=True
        ),
        "repeat_id": repeat_id,
        "order_id": order_id,
        "timestamp_utc": _utc_now(),
    }


def run_pointwise(
    providers: Sequence[JudgeProvider],
    images: pd.DataFrame,
    input_root: Path,
    store: JudgeRunStore,
    repeats: int = 1,
    attempts: int = 3,
    initial_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Evaluate each unique image while preserving successful checkpoints."""

    required = {"dataset_id", "imagem_id", "content_sha256", "caminho_relativo"}
    missing = required - set(images)
    if missing:
        raise ValueError(f"Pointwise inventory is missing columns: {sorted(missing)}")
    if images["imagem_id"].duplicated().any():
        raise ValueError("Pointwise judging requires unique imagem_id values")
    if repeats < 1:
        raise ValueError("repeats must be at least one")

    for provider in providers:
        provider.ensure_available()
        for row in images.itertuples(index=False):
            path = Path(input_root) / row.caminho_relativo
            for repeat_id in range(1, repeats + 1):
                key_payload = {
                    "task": "pointwise",
                    "provider": provider.provider_name,
                    "model": provider.model_id,
                    "revision": provider.model_revision,
                    "inference_config": _inference_config(provider),
                    "prompt": PROMPT_VERSION,
                    "dataset_id": row.dataset_id,
                    "imagem_id": row.imagem_id,
                    "image": row.content_sha256,
                    "repeat": repeat_id,
                }
                task_key = _stable_hash(key_payload)
                if store.is_complete(task_key):
                    continue
                raw, error, used_attempts = _call_with_retry(
                    lambda: provider.evaluate(POINTWISE_PROMPT, [path]),
                    attempts,
                    initial_delay,
                    sleep,
                )
                record = {
                    **_base_record(provider, task_key, "pointwise", repeat_id, None),
                    "dataset_id": row.dataset_id,
                    "imagem_id": row.imagem_id,
                    "content_sha256": row.content_sha256,
                    "attempts": used_attempts,
                }
                if raw is None:
                    store.append({**record, "status": "error", "error": error})
                    _raise_if_quota_exhausted(provider, error)
                    continue
                try:
                    parsed = PointwiseJudgment.parse(raw)
                except (ValueError, json.JSONDecodeError) as exc:
                    store.append(
                        {
                            **record,
                            "status": "parse_error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "raw_response": raw,
                            "raw_response_sha256": hashlib.sha256(
                                raw.encode("utf-8")
                            ).hexdigest(),
                        }
                    )
                    continue
                store.append(
                    {
                        **record,
                        "status": "success",
                        "raw_response_sha256": hashlib.sha256(
                            raw.encode("utf-8")
                        ).hexdigest(),
                        **parsed.to_dict(),
                    }
                )


def build_pair_tasks(images: pd.DataFrame, input_root: Path) -> list[PairTask]:
    """Build both A/B orders for every complete neutral-inclusive pair."""

    required = {
        "dataset_id",
        "pair_id",
        "tipo_prompt",
        "imagem_id",
        "content_sha256",
        "caminho_relativo",
    }
    missing = required - set(images)
    if missing:
        raise ValueError(f"Pair inventory is missing columns: {sorted(missing)}")
    data = images.copy()
    data["prompt_group"] = data["tipo_prompt"].map(normalize_prompt_group)
    tasks = []
    for (dataset_id, pair_id), group in data.groupby(["dataset_id", "pair_id"]):
        if len(group) != 2 or set(group["prompt_group"]) != {"neutral", "inclusive"}:
            raise ValueError(f"Incomplete or ambiguous pair: {dataset_id}/{pair_id}")
        neutral = group.loc[group["prompt_group"] == "neutral"].iloc[0]
        inclusive = group.loc[group["prompt_group"] == "inclusive"].iloc[0]
        orders = ((neutral, inclusive, "neutral-first"), (inclusive, neutral, "inclusive-first"))
        for first, second, order_id in orders:
            tasks.append(
                PairTask(
                    dataset_id=str(dataset_id),
                    pair_id=str(pair_id),
                    image_a_id=str(first["imagem_id"]),
                    image_b_id=str(second["imagem_id"]),
                    image_a_path=Path(input_root) / first["caminho_relativo"],
                    image_b_path=Path(input_root) / second["caminho_relativo"],
                    image_a_hash=str(first["content_sha256"]),
                    image_b_hash=str(second["content_sha256"]),
                    order_id=order_id,
                )
            )
    return tasks


def run_pairwise(
    providers: Sequence[JudgeProvider],
    tasks: Iterable[PairTask],
    store: JudgeRunStore,
    attempts: int = 3,
    initial_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Evaluate matched pairs in both orders and normalize positional choices."""

    task_list = list(tasks)
    for provider in providers:
        provider.ensure_available()
        for task in task_list:
            key_payload = {
                "task": "pairwise",
                "provider": provider.provider_name,
                "model": provider.model_id,
                "revision": provider.model_revision,
                "inference_config": _inference_config(provider),
                "prompt": PROMPT_VERSION,
                "dataset_id": task.dataset_id,
                "pair_id": task.pair_id,
                "images": [task.image_a_hash, task.image_b_hash],
                "order": task.order_id,
            }
            task_key = _stable_hash(key_payload)
            if store.is_complete(task_key):
                continue
            raw, error, used_attempts = _call_with_retry(
                lambda: provider.evaluate(
                    PAIRWISE_PROMPT, [task.image_a_path, task.image_b_path]
                ),
                attempts,
                initial_delay,
                sleep,
            )
            record = {
                **_base_record(provider, task_key, "pairwise", 1, task.order_id),
                "dataset_id": task.dataset_id,
                "pair_id": task.pair_id,
                "image_a_id": task.image_a_id,
                "image_b_id": task.image_b_id,
                "image_a_sha256": task.image_a_hash,
                "image_b_sha256": task.image_b_hash,
                "attempts": used_attempts,
            }
            if raw is None:
                store.append({**record, "status": "error", "error": error})
                _raise_if_quota_exhausted(provider, error)
                continue
            try:
                parsed = PairwiseJudgment.parse(raw)
            except (ValueError, json.JSONDecodeError) as exc:
                store.append(
                    {
                        **record,
                        "status": "parse_error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "raw_response": raw,
                        "raw_response_sha256": hashlib.sha256(
                            raw.encode("utf-8")
                        ).hexdigest(),
                    }
                )
                continue
            store.append(
                {
                    **record,
                    "status": "success",
                    "raw_response_sha256": hashlib.sha256(
                        raw.encode("utf-8")
                    ).hexdigest(),
                    **parsed.to_dict(),
                    "racial_winner_id": normalize_pair_choice(
                        parsed.racial_choice, task.image_a_id, task.image_b_id
                    ),
                    "gender_winner_id": normalize_pair_choice(
                        parsed.gender_choice, task.image_a_id, task.image_b_id
                    ),
                }
            )


def export_judge_tables(store: JudgeRunStore, output_dir: Path) -> tuple[Path, Path]:
    """Export successful pointwise and pairwise records to stable CSV tables."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = pd.DataFrame(store.records())
    if records.empty:
        successful = pd.DataFrame(columns=["task_type"])
    elif "status" not in records or "task_type" not in records:
        raise ValueError("Judge run store is missing status or task_type")
    else:
        successful = records.loc[records["status"] == "success"]
    pointwise = successful.loc[successful["task_type"] == "pointwise"].copy()
    pairwise = successful.loc[successful["task_type"] == "pairwise"].copy()
    pointwise_columns = [
        "task_key", "dataset_id", "imagem_id", "content_sha256", "provider",
        "model_id", "model_revision", "prompt_version", "temperature", "seed",
        "inference_config_json", "repeat_id", "timestamp_utc", "attempts", "eligible", "racial_score",
        "gender_score", "abstain_reason", "raw_response_sha256",
    ]
    pairwise_columns = [
        "task_key", "dataset_id", "pair_id", "provider", "model_id",
        "model_revision", "prompt_version", "temperature", "seed", "order_id",
        "inference_config_json", "timestamp_utc", "attempts", "image_a_id", "image_b_id",
        "image_a_sha256", "image_b_sha256", "racial_choice", "gender_choice",
        "racial_winner_id", "gender_winner_id", "raw_response_sha256",
    ]
    pointwise = pointwise.reindex(columns=pointwise_columns)
    pairwise = pairwise.reindex(columns=pairwise_columns)
    pointwise_path = output_dir / "judge_pointwise.csv"
    pairwise_path = output_dir / "judge_pairwise.csv"
    pointwise.to_csv(pointwise_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)
    return pointwise_path, pairwise_path
