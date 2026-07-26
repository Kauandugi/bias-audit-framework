"""L2-normalized CLIP prototype scoring for perceived diversity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from .constants import CLIP_OUTPUT_COLUMNS

DEFAULT_PROTOTYPE_TEXTS: dict[str, tuple[str, ...]] = {
    "racial_diversity": (
        "a racially diverse group of people",
        "people with diverse racial appearances",
        "a group with varied skin tones and racialized appearances",
    ),
    "racial_homogeneity": (
        "a racially homogeneous group of people",
        "people with similar racial appearances",
        "a group with little variation in racialized appearance",
    ),
    "gender_diversity": (
        "a gender-diverse group of people",
        "people with diverse gender presentation",
        "a group with varied masculine, feminine, and androgynous presentation",
    ),
    "gender_homogeneity": (
        "a gender-homogeneous group of people",
        "people with similar gender presentation",
        "a group with one visually similar gender presentation",
    ),
}

def extract_clip_feature_tensor(output: Any) -> torch.Tensor:
    """Support tensor and Transformers v4/v5 feature return types."""

    if isinstance(output, torch.Tensor):
        return output
    pooler_output = getattr(output, "pooler_output", None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(
        "Unexpected CLIP output; expected Tensor or object with Tensor pooler_output"
    )


def cosine_scores(
    image_features: torch.Tensor, text_features: torch.Tensor
) -> torch.Tensor:
    """Calculate pure cosine similarity through explicit L2 normalization."""

    image_unit = F.normalize(image_features, p=2, dim=-1)
    text_unit = F.normalize(text_features, p=2, dim=-1)
    return image_unit @ text_unit.T


def build_prototypes(
    text_features: torch.Tensor,
    group_sizes: Sequence[int],
) -> torch.Tensor:
    """Average unit phrase vectors per group and renormalize each prototype."""

    if sum(group_sizes) != len(text_features):
        raise ValueError("group_sizes do not match text feature count")
    if not group_sizes or any(size <= 0 for size in group_sizes):
        raise ValueError("Every prototype group must contain at least one phrase")
    phrase_units = F.normalize(text_features, p=2, dim=-1)
    prototypes = []
    offset = 0
    for size in group_sizes:
        prototypes.append(phrase_units[offset : offset + size].mean(dim=0))
        offset += size
    return F.normalize(torch.stack(prototypes), p=2, dim=-1)


class ClipDiversityScorer:
    """Cache text prototypes and score each complete image exactly once."""

    def __init__(
        self,
        model_id: str,
        device: torch.device,
        prototype_texts: Mapping[str, Sequence[str]] = DEFAULT_PROTOTYPE_TEXTS,
    ) -> None:
        from transformers import CLIPModel, CLIPProcessor

        expected_keys = tuple(DEFAULT_PROTOTYPE_TEXTS)
        if tuple(prototype_texts) != expected_keys:
            raise ValueError(f"Prototype keys must be exactly {expected_keys}")
        self.device = device
        self.prototype_texts = {
            key: tuple(values) for key, values in prototype_texts.items()
        }
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.prototypes = self._encode_prototypes()

    def _encode_prototypes(self) -> torch.Tensor:
        phrases = [
            phrase for values in self.prototype_texts.values() for phrase in values
        ]
        inputs = self.processor(text=phrases, return_tensors="pt", padding=True)
        text_inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if key in {"input_ids", "attention_mask"}
        }
        with torch.inference_mode():
            output = self.model.get_text_features(**text_inputs)
            features = extract_clip_feature_tensor(output)
        return build_prototypes(
            features, [len(values) for values in self.prototype_texts.values()]
        )

    def __call__(self, image_path: Path) -> dict[str, float]:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        image_inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if key == "pixel_values"
        }
        with torch.inference_mode():
            output = self.model.get_image_features(**image_inputs)
            features = extract_clip_feature_tensor(output)
            scores = cosine_scores(features, self.prototypes)[0].detach().cpu()
        racial_diverse, racial_homogeneous, gender_diverse, gender_homogeneous = (
            map(float, scores)
        )
        return {
            "clip_racial_diversity_cosine": racial_diverse,
            "clip_racial_homogeneity_cosine": racial_homogeneous,
            "clip_racial_diversity_margin": racial_diverse - racial_homogeneous,
            "clip_gender_diversity_cosine": gender_diverse,
            "clip_gender_homogeneity_cosine": gender_homogeneous,
            "clip_gender_diversity_margin": gender_diverse - gender_homogeneous,
        }
