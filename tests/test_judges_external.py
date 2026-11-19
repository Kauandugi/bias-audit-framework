from __future__ import annotations

import os

import pytest

from biasauditfw.judges.providers import (
    GeminiProvider,
    GroqProvider,
    LocalGemmaProvider,
    LocalQwen3VLProvider,
    LocalQwenProvider,
)


@pytest.mark.external
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY is not set")
def test_gemini_model_preflight() -> None:
    GeminiProvider().ensure_available()


@pytest.mark.external
@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY is not set")
def test_groq_model_preflight() -> None:
    GroqProvider().ensure_available()


@pytest.mark.external
@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("BIASAUDIT_RUN_LOCAL_QWEN") != "1",
    reason="BIASAUDIT_RUN_LOCAL_QWEN is not enabled",
)
def test_local_qwen_model_preflight() -> None:
    LocalQwenProvider().ensure_available()


@pytest.mark.external
@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("BIASAUDIT_RUN_LOCAL_QWEN3") != "1",
    reason="BIASAUDIT_RUN_LOCAL_QWEN3 is not enabled",
)
def test_local_qwen3_model_preflight() -> None:
    LocalQwen3VLProvider().ensure_available()


@pytest.mark.external
@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("BIASAUDIT_RUN_LOCAL_GEMMA") != "1",
    reason="BIASAUDIT_RUN_LOCAL_GEMMA is not enabled",
)
def test_local_gemma_model_preflight() -> None:
    LocalGemmaProvider().ensure_available()
