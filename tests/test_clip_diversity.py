from __future__ import annotations

import pytest
import torch

from biasauditfw.clip_diversity import (
    DEFAULT_PROTOTYPE_TEXTS,
    build_prototypes,
    cosine_scores,
    extract_clip_feature_tensor,
)


def test_prompt_families_have_expected_order_and_multiple_phrases() -> None:
    assert tuple(DEFAULT_PROTOTYPE_TEXTS) == (
        "racial_diversity",
        "racial_homogeneity",
        "gender_diversity",
        "gender_homogeneity",
    )
    assert all(len(phrases) == 3 for phrases in DEFAULT_PROTOTYPE_TEXTS.values())


def test_cosine_scores_and_prototype_renormalization() -> None:
    text = torch.tensor(
        [[2.0, 0.0], [1.0, 0.0], [0.0, 3.0], [0.0, 1.0]]
    )
    prototypes = build_prototypes(text, [2, 2])
    scores = cosine_scores(torch.tensor([[4.0, 0.0]]), prototypes)

    assert torch.allclose(torch.linalg.vector_norm(prototypes, dim=1), torch.ones(2))
    assert scores.tolist()[0] == pytest.approx([1.0, 0.0])
    assert torch.all((scores >= -1) & (scores <= 1))


def test_invalid_prototype_sizes_and_feature_outputs() -> None:
    tensor = torch.tensor([[1.0, 2.0]])
    assert extract_clip_feature_tensor(tensor) is tensor

    class Output:
        pooler_output = tensor

    assert extract_clip_feature_tensor(Output()) is tensor
    assert extract_clip_feature_tensor((tensor,)) is tensor
    with pytest.raises(TypeError, match="Unexpected CLIP output"):
        extract_clip_feature_tensor(object())
    with pytest.raises(ValueError, match="do not match"):
        build_prototypes(tensor, [2])
    with pytest.raises(ValueError, match="at least one"):
        build_prototypes(tensor, [0, 1])

