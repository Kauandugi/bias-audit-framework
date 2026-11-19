from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pytest

from biasauditfw.judges.contracts import (
    PairwiseJudgment,
    PointwiseJudgment,
    normalize_pair_choice,
)
from biasauditfw.judges.runner import (
    JudgeRunStore,
    build_pair_tasks,
    export_judge_tables,
    run_pairwise,
    run_pointwise,
)
from biasauditfw.judges.providers import (
    GroqProvider,
    LocalGemmaProvider,
    LocalQwen3VLProvider,
    LocalQwenProvider,
)

from conftest import make_image


class FakeProvider:
    provider_name = "fake"
    model_id = "fake-vision"
    model_revision = "test-revision"

    def __init__(self, failures: int = 0) -> None:
        self.calls = 0
        self.preflights = 0
        self.failures = failures

    def ensure_available(self) -> None:
        self.preflights += 1

    def evaluate(self, prompt: str, image_paths: list[Path]) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("temporary rate limit")
        if len(image_paths) == 1:
            return json.dumps(
                {
                    "eligible": True,
                    "racial_score": 4,
                    "gender_score": 3,
                    "abstain_reason": None,
                }
            )
        inclusive_index = next(
            index for index, path in enumerate(image_paths) if "inclusive" in path.name
        )
        choice = "A" if inclusive_index == 0 else "B"
        return json.dumps({"racial_choice": choice, "gender_choice": choice})


class QuotaExhaustedProvider(FakeProvider):
    def evaluate(self, prompt: str, image_paths: list[Path]) -> str:
        self.calls += 1
        raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")


def inventory(tmp_path: Path) -> pd.DataFrame:
    neutral = make_image(tmp_path / "neutral.png", (10, 20, 30))
    inclusive = make_image(tmp_path / "inclusive.png", (40, 50, 60))
    return pd.DataFrame(
        [
            {
                "dataset_id": "corpus-a",
                "imagem_id": "neutral-id",
                "content_sha256": "a" * 64,
                "caminho_relativo": neutral.name,
                "pair_id": "p1",
                "tipo_prompt": "Neutral Prompts",
            },
            {
                "dataset_id": "corpus-a",
                "imagem_id": "inclusive-id",
                "content_sha256": "b" * 64,
                "caminho_relativo": inclusive.name,
                "pair_id": "p1",
                "tipo_prompt": "Inclusive Prompts",
            },
        ]
    )


def test_judgment_contracts_are_strict() -> None:
    parsed = PointwiseJudgment.parse(
        "```json\n{\"eligible\": true, \"racial_score\": 4, "
        "\"gender_score\": 2, \"abstain_reason\": null}\n```"
    )
    assert parsed.racial_score == 4
    assert PointwiseJudgment.parse(
        {
            "eligible": False,
            "racial_score": None,
            "gender_score": None,
            "abstain_reason": "one person",
        }
    ).eligible is False
    with pytest.raises(ValueError, match="1 to 5"):
        PointwiseJudgment.parse(
            {
                "eligible": True,
                "racial_score": 6,
                "gender_score": 2,
                "abstain_reason": None,
            }
        )
    with pytest.raises(ValueError, match="abstain_reason"):
        PointwiseJudgment.parse(
            {
                "eligible": False,
                "racial_score": None,
                "gender_score": None,
                "abstain_reason": None,
            }
        )
    pair = PairwiseJudgment.parse(
        {"racial_choice": "A", "gender_choice": "tie"}
    )
    assert pair.gender_choice == "tie"
    assert normalize_pair_choice("B", "first", "second") == "second"


def test_runner_retries_resumes_and_exports(tmp_path: Path) -> None:
    images = inventory(tmp_path)
    store = JudgeRunStore(tmp_path / "judge_runs.jsonl")
    provider = FakeProvider(failures=1)
    run_pointwise(
        [provider],
        images,
        tmp_path,
        store,
        repeats=2,
        initial_delay=0,
        sleep=lambda _: None,
    )
    assert provider.calls == 5
    assert len([row for row in store.records() if row["status"] == "success"]) == 4
    run_pointwise([provider], images, tmp_path, store, repeats=2)
    assert provider.calls == 5

    tasks = build_pair_tasks(images, tmp_path)
    assert {task.order_id for task in tasks} == {"neutral-first", "inclusive-first"}
    run_pairwise([provider], tasks, store, initial_delay=0, sleep=lambda _: None)
    pair_records = [row for row in store.records() if row["task_type"] == "pairwise"]
    assert len(pair_records) == 2
    assert {row["racial_winner_id"] for row in pair_records} == {"inclusive-id"}

    pointwise_path, pairwise_path = export_judge_tables(store, tmp_path / "tables")
    assert len(pd.read_csv(pointwise_path)) == 4
    assert len(pd.read_csv(pairwise_path)) == 2


def test_runner_stops_provider_after_persistent_quota_error(tmp_path: Path) -> None:
    images = inventory(tmp_path)
    store = JudgeRunStore(tmp_path / "judge_runs.jsonl")
    provider = QuotaExhaustedProvider()

    with pytest.raises(RuntimeError, match="exhausted its quota"):
        run_pointwise(
            [provider],
            images,
            tmp_path,
            store,
            attempts=3,
            initial_delay=0,
            sleep=lambda _: None,
        )

    assert provider.calls == 3
    records = store.records()
    assert len(records) == 1
    assert records[0]["status"] == "error"


def test_store_rejects_invalid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSONL"):
        JudgeRunStore(path)


def test_groq_uses_explicit_schema_and_disables_reasoning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = make_image(tmp_path / "group.png", (10, 20, 30))
    captured: dict[str, object] = {}

    def create(**kwargs: object) -> object:
        captured.update(kwargs)
        message = SimpleNamespace(
            content=json.dumps(
                {
                    "eligible": True,
                    "racial_score": 3,
                    "gender_score": 4,
                    "abstain_reason": None,
                }
            )
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setitem(
        sys.modules,
        "groq",
        SimpleNamespace(Groq=lambda **_: fake_client),
    )
    provider = GroqProvider(api_key="test-key")
    raw = provider.evaluate("Return JSON", [image])

    assert provider.model_id == "qwen/qwen3.8-27b"
    assert PointwiseJudgment.parse(raw).gender_score == 4
    assert captured["reasoning_effort"] == "none"
    response_format = captured["response_format"]
    assert isinstance(response_format, dict)
    assert response_format["type"] == "json_schema"
    schema = response_format["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "eligible",
        "racial_score",
        "gender_score",
        "abstain_reason",
    }


def test_groq_configures_request_timeout_and_disables_sdk_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def build_client(**kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=build_client))
    provider = GroqProvider(api_key="test-key", request_timeout_seconds=12.5)

    assert provider.request_timeout_seconds == 12.5
    assert captured["timeout"] == 12.5
    assert captured["max_retries"] == 0
    with pytest.raises(ValueError, match="positive"):
        GroqProvider(api_key="test-key", request_timeout_seconds=0)


def test_groq_pairwise_uses_client_side_json_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = make_image(tmp_path / "first.png", (10, 20, 30))
    second = make_image(tmp_path / "second.png", (40, 50, 60))
    captured: dict[str, object] = {}

    def create(**kwargs: object) -> object:
        captured.update(kwargs)
        message = SimpleNamespace(
            content=json.dumps(
                {"racial_choice": "A", "gender_choice": "tie"}
            )
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setitem(
        sys.modules,
        "groq",
        SimpleNamespace(Groq=lambda **_: fake_client),
    )
    raw = GroqProvider(api_key="test-key").evaluate(
        "Return only JSON", [first, second]
    )

    assert PairwiseJudgment.parse(raw).gender_choice == "tie"
    assert "response_format" not in captured
    assert captured["reasoning_effort"] == "none"


def test_local_qwen_limits_visual_tokens_and_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            captured["processor_args"] = args
            captured["processor_kwargs"] = kwargs
            return object()

    fake_pipeline = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(_commit_hash="resolved-revision"))
    )

    def pipeline(*args: object, **kwargs: object) -> object:
        captured["pipeline_args"] = args
        captured["pipeline_kwargs"] = kwargs
        return fake_pipeline

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs: object) -> None:
            captured["quantization_kwargs"] = kwargs

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        float16="float16",
    )

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=FakeProcessor,
            BitsAndBytesConfig=FakeBitsAndBytesConfig,
            pipeline=pipeline,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    provider = LocalQwenProvider()
    provider.ensure_available()

    processor_kwargs = captured["processor_kwargs"]
    assert isinstance(processor_kwargs, dict)
    assert processor_kwargs["min_pixels"] == 256 * 28 * 28
    assert processor_kwargs["max_pixels"] == 768 * 28 * 28
    assert provider.inference_config["max_new_tokens"] == 128
    assert provider.inference_config["quantization"] == "4bit-nf4"
    quantization_kwargs = captured["quantization_kwargs"]
    assert isinstance(quantization_kwargs, dict)
    assert quantization_kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "float16",
    }
    pipeline_kwargs = captured["pipeline_kwargs"]
    assert isinstance(pipeline_kwargs, dict)
    assert "quantization_config" in pipeline_kwargs["model_kwargs"]
    assert provider.model_revision == "resolved-revision"


def test_local_qwen_rejects_unknown_quantization() -> None:
    with pytest.raises(ValueError, match="quantization"):
        LocalQwenProvider(quantization="silent-fallback")


def test_new_local_judges_have_explicit_models_and_independent_ids() -> None:
    qwen = LocalQwen3VLProvider()
    gemma = LocalGemmaProvider()

    assert qwen.model_id == "Qwen/Qwen3-VL-8B-Instruct"
    assert qwen.provider_name == "local-qwen3-vl"
    assert gemma.model_id == "google/gemma-3-4b-it"
    assert gemma.provider_name == "local-gemma3"
    assert qwen.inference_config["quantization"] == "4bit-nf4"
    assert gemma.inference_config["compute_dtype"] == "bfloat16"
    assert qwen.inference_config["return_full_text"] is False
    assert gemma.inference_config["return_full_text"] is True


def test_local_gemma_uses_bfloat16_for_nf4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            return object()

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs: object) -> None:
            captured["quantization_kwargs"] = kwargs

    fake_pipeline = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(_commit_hash=None))
    )

    def pipeline(*args: object, **kwargs: object) -> object:
        captured["model_kwargs"] = kwargs["model_kwargs"]
        return fake_pipeline

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        float16="float16",
        bfloat16="bfloat16",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=FakeProcessor,
            BitsAndBytesConfig=FakeBitsAndBytesConfig,
            pipeline=pipeline,
        ),
    )

    LocalGemmaProvider().ensure_available()

    assert captured["quantization_kwargs"] == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",
    }
    assert captured["model_kwargs"]["dtype"] == "bfloat16"


def test_local_judge_extracts_nested_hf_output_and_orders_images(
    tmp_path: Path,
) -> None:
    first = make_image(tmp_path / "first.png", (10, 20, 30))
    second = make_image(tmp_path / "second.png", (40, 50, 60))
    captured: dict[str, object] = {}

    def fake_pipeline(**kwargs: object) -> list[dict[str, object]]:
        captured.update(kwargs)
        return [
            {
                "generated_text": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": '{"racial_choice":"A","gender_choice":"B"}',
                            }
                        ],
                    }
                ]
            }
        ]

    provider = LocalGemmaProvider(quantization="none")
    provider._pipeline = fake_pipeline
    raw = provider.evaluate("Return JSON", [first, second])

    assert PairwiseJudgment.parse(raw).gender_choice == "B"
    messages = captured["text"]
    assert isinstance(messages, list)
    content = messages[0]["content"]
    assert [item.get("text") for item in content if item["type"] == "text"] == [
        "Image A:",
        "Image B:",
        "Return JSON",
    ]
    assert captured["max_new_tokens"] == 128
    assert captured["generate_kwargs"] == {"do_sample": False}
    assert captured["return_full_text"] is True
    image_items = [item for item in content if item["type"] == "image"]
    assert len(image_items) == 2
    assert [item["url"] for item in image_items] == [str(first), str(second)]
    with pytest.raises(ValueError, match="one or two"):
        provider.evaluate("Return JSON", [])


def test_local_judge_rejects_missing_image_before_inference(tmp_path: Path) -> None:
    provider = LocalGemmaProvider(quantization="none")
    provider._pipeline = lambda **kwargs: []

    with pytest.raises(FileNotFoundError):
        provider.evaluate("Return JSON", [tmp_path / "missing.png"])


def test_prompts_do_not_anchor_scores_or_pairwise_positions() -> None:
    from biasauditfw.judges.prompts import (
        PAIRWISE_PROMPT,
        POINTWISE_PROMPT,
        PROMPT_VERSION,
    )

    assert PROMPT_VERSION == "mllm-diversity-v5"
    assert '"racial_choice": "A"' not in PAIRWISE_PROMPT
    assert '"gender_choice": "B"' not in PAIRWISE_PROMPT
    assert '"racial_score": 1' not in POINTWISE_PROMPT
    assert '"gender_score": 1' not in POINTWISE_PROMPT
