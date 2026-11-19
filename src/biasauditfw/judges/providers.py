"""Provider adapters for cloud and local multimodal judges."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Protocol, Sequence

from PIL import Image


def _groq_response_format(image_count: int) -> dict[str, object]:
    """Return the response schema matching the pointwise or pairwise prompt."""

    if image_count == 1:
        properties: dict[str, object] = {
            "eligible": {"type": "boolean"},
            "racial_score": {"type": ["integer", "null"], "minimum": 1, "maximum": 5},
            "gender_score": {"type": ["integer", "null"], "minimum": 1, "maximum": 5},
            "abstain_reason": {"type": ["string", "null"]},
        }
        required = ["eligible", "racial_score", "gender_score", "abstain_reason"]
        name = "pointwise_diversity_judgment"
    else:
        raise ValueError("Groq structured output is only used for pointwise judging")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": False,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


class JudgeProvider(Protocol):
    """Interface used by the deterministic runner and test doubles."""

    provider_name: str
    model_id: str
    model_revision: str

    def ensure_available(self) -> None:
        ...

    def evaluate(self, prompt: str, image_paths: Sequence[Path]) -> str:
        ...


class GeminiProvider:
    provider_name = "google"
    seed = 42

    def __init__(
        self,
        model_id: str = "gemini-3.6-flash",
        api_key: str | None = None,
        request_timeout_seconds: float = 60.0,
    ) -> None:
        from google import genai

        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is required for the Gemini judge")
        self.model_id = model_id
        self.model_revision = model_id
        self.request_timeout_seconds = float(request_timeout_seconds)
        self._types = __import__("google.genai.types", fromlist=["types"])
        self._client = genai.Client(
            api_key=key,
            http_options=self._types.HttpOptions(
                timeout=int(self.request_timeout_seconds * 1000),
                retryOptions=self._types.HttpRetryOptions(attempts=1),
            ),
        )

    def ensure_available(self) -> None:
        model = self._client.models.get(model=self.model_id)
        self.model_revision = str(
            getattr(model, "version", None) or getattr(model, "name", self.model_id)
        )

    def evaluate(self, prompt: str, image_paths: Sequence[Path]) -> str:
        parts = [prompt]
        for index, path in enumerate(image_paths):
            if len(image_paths) == 2:
                parts.append(f"Image {'A' if index == 0 else 'B'}:")
            with Image.open(path) as image:
                mime = Image.MIME.get(image.format, "image/png")
            parts.append(
                self._types.Part.from_bytes(data=path.read_bytes(), mime_type=mime)
            )
        response = self._client.models.generate_content(
            model=self.model_id,
            contents=parts,
            config=self._types.GenerateContentConfig(
                temperature=0,
                seed=self.seed,
                response_mime_type="application/json",
            ),
        )
        if not response.text:
            raise RuntimeError("Gemini returned an empty response")
        return response.text


class GroqProvider:
    provider_name = "groq"
    seed = 42

    def __init__(
        self,
        model_id: str = "qwen/qwen3.8-27b",
        api_key: str | None = None,
        request_timeout_seconds: float = 60.0,
    ) -> None:
        from groq import Groq

        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is required for the Groq judge")
        self.model_id = model_id
        self.model_revision = model_id
        self.request_timeout_seconds = float(request_timeout_seconds)
        self._client = Groq(
            api_key=key,
            timeout=self.request_timeout_seconds,
            max_retries=0,
        )

    def ensure_available(self) -> None:
        model = self._client.models.retrieve(self.model_id)
        self.model_revision = str(getattr(model, "id", self.model_id))

    def evaluate(self, prompt: str, image_paths: Sequence[Path]) -> str:
        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        for index, path in enumerate(image_paths):
            if len(image_paths) == 2:
                content.append(
                    {"type": "text", "text": f"Image {'A' if index == 0 else 'B'}:"}
                )
            with Image.open(path) as image:
                mime = Image.MIME.get(image.format, "image/png")
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded}"},
                }
            )
        request: dict[str, object] = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "seed": self.seed,
            "reasoning_effort": "none",
            "max_completion_tokens": 512,
        }
        if len(image_paths) == 1:
            request["response_format"] = _groq_response_format(1)
        elif len(image_paths) != 2:
            raise ValueError("Groq judging requires one or two images")
        response = self._client.chat.completions.create(
            **request,
        )
        result = response.choices[0].message.content
        if not result:
            raise RuntimeError("Groq returned an empty response")
        return result


def _generated_text(value: Any) -> str:
    """Extract assistant text from the shapes returned by HF pipelines."""

    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str) and text.strip():
            return text
        return _generated_text(value.get("content"))
    if isinstance(value, list):
        for item in reversed(value):
            text = _generated_text(item)
            if text.strip():
                return text
    return ""


class _LocalTransformersVisionProvider:
    """Shared deterministic adapter for local image-text Transformers models."""

    provider_name = "local-huggingface"
    seed = 42

    def __init__(
        self,
        model_id: str,
        model_revision: str = "main",
        max_new_tokens: int = 128,
        quantization: str = "4bit-nf4",
        processor_kwargs: dict[str, object] | None = None,
        return_full_text: bool = False,
        compute_dtype: str = "float16",
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("Local judge max_new_tokens must be positive")
        if quantization not in {"none", "4bit-nf4"}:
            raise ValueError("Local judge quantization must be 'none' or '4bit-nf4'")
        if compute_dtype not in {"float16", "bfloat16"}:
            raise ValueError("Local judge compute_dtype must be float16 or bfloat16")
        self.model_id = model_id
        self.model_revision = model_revision
        self.max_new_tokens = int(max_new_tokens)
        self.quantization = quantization
        self.processor_kwargs = dict(processor_kwargs or {})
        self.return_full_text = bool(return_full_text)
        self.compute_dtype = compute_dtype
        self._pipeline = None

    @property
    def inference_config(self) -> dict[str, object]:
        config: dict[str, object] = {
            "max_new_tokens": self.max_new_tokens,
            "quantization": self.quantization,
            "compute_dtype": (
                self.compute_dtype if self.quantization == "4bit-nf4" else "auto"
            ),
            "return_full_text": self.return_full_text,
        }
        if self.processor_kwargs:
            config["processor_kwargs"] = self.processor_kwargs
        return config

    def ensure_available(self) -> None:
        if self._pipeline is not None:
            return
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, pipeline

        model_kwargs: dict[str, object] = {"dtype": "auto"}
        if self.quantization == "4bit-nf4":
            if not torch.cuda.is_available():
                raise RuntimeError("4-bit local judge inference requires a CUDA GPU")
            compute_dtype = getattr(torch, self.compute_dtype)
            model_kwargs = {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                ),
                "dtype": compute_dtype,
            }

        processor = AutoProcessor.from_pretrained(
            self.model_id,
            revision=self.model_revision,
            **self.processor_kwargs,
        )

        self._pipeline = pipeline(
            "image-text-to-text",
            model=self.model_id,
            revision=self.model_revision,
            device_map="auto",
            model_kwargs=model_kwargs,
            processor=processor,
        )
        resolved = getattr(self._pipeline.model.config, "_commit_hash", None)
        if resolved:
            self.model_revision = str(resolved)

    def evaluate(self, prompt: str, image_paths: Sequence[Path]) -> str:
        self.ensure_available()
        if len(image_paths) not in {1, 2}:
            raise ValueError("Local judging requires one or two images")
        content: list[dict[str, object]] = []
        for index, path in enumerate(image_paths):
            if not path.is_file():
                raise FileNotFoundError(path)
            if len(image_paths) == 2:
                content.append(
                    {"type": "text", "text": f"Image {'A' if index == 0 else 'B'}:"}
                )
            content.append({"type": "image", "url": str(path)})
        content.append({"type": "text", "text": prompt})
        output = self._pipeline(
            text=[{"role": "user", "content": content}],
            max_new_tokens=self.max_new_tokens,
            generate_kwargs={"do_sample": False},
            return_full_text=self.return_full_text,
        )
        generated = _generated_text(output[0]["generated_text"])
        if not generated:
            raise RuntimeError("Local multimodal judge returned an empty response")
        return generated


class LocalQwenProvider(_LocalTransformersVisionProvider):
    """Legacy Qwen2.5-VL adapter retained for reproducibility."""

    provider_name = "local-qwen2.5-vl"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        model_revision: str = "main",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 768 * 28 * 28,
        max_new_tokens: int = 128,
        quantization: str = "4bit-nf4",
    ) -> None:
        if min_pixels <= 0 or max_pixels < min_pixels:
            raise ValueError("Qwen pixel limits must satisfy 0 < min <= max")
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        super().__init__(
            model_id=model_id,
            model_revision=model_revision,
            max_new_tokens=max_new_tokens,
            quantization=quantization,
            processor_kwargs={
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
            },
        )

    @property
    def inference_config(self) -> dict[str, object]:
        config = super().inference_config
        config.pop("processor_kwargs", None)
        config.update(
            {
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
            }
        )
        return config


class LocalQwen3VLProvider(_LocalTransformersVisionProvider):
    """Qwen3-VL local judge for the gated Colab/T4 experiment."""

    provider_name = "local-qwen3-vl"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        model_revision: str = "main",
        max_new_tokens: int = 128,
        quantization: str = "4bit-nf4",
    ) -> None:
        super().__init__(
            model_id=model_id,
            model_revision=model_revision,
            max_new_tokens=max_new_tokens,
            quantization=quantization,
        )


class LocalGemmaProvider(_LocalTransformersVisionProvider):
    """Gemma 3 local judge from a model family independent of Qwen."""

    provider_name = "local-gemma3"

    def __init__(
        self,
        model_id: str = "google/gemma-3-4b-it",
        model_revision: str = "main",
        max_new_tokens: int = 128,
        quantization: str = "4bit-nf4",
    ) -> None:
        super().__init__(
            model_id=model_id,
            model_revision=model_revision,
            max_new_tokens=max_new_tokens,
            quantization=quantization,
            return_full_text=True,
            compute_dtype="bfloat16",
        )
