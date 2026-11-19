from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_tcc_release_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("build_tcc_release_checkpoint", SCRIPT)
assert SPEC and SPEC.loader
checkpoint_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checkpoint_module)


def test_public_checkpoint_contract_excludes_private_paths() -> None:
    assert checkpoint_module.PUBLIC_ARTIFACTS
    for path in checkpoint_module.PUBLIC_ARTIFACTS:
        normalized = path.replace("\\", "/")
        assert not normalized.startswith(checkpoint_module.EXCLUDED_PRIVATE_PREFIXES)
    human_paths = {
        path
        for path in checkpoint_module.PUBLIC_ARTIFACTS
        if path.startswith("data/human-unesp-aggregate/")
    }
    assert human_paths == {
        "data/human-unesp-aggregate/human_ratings_aggregate.csv",
        "data/human-unesp-aggregate/human_interrater_agreement.csv",
        "data/human-unesp-aggregate/human_paired_prompt_effects.csv",
        "data/human-unesp-aggregate/human_participant_sensitivity.csv",
        "data/human-unesp-aggregate/human_framework_convergence.csv",
        "data/human-unesp-aggregate/deepface_simpson_paired.csv",
        "data/human-unesp-aggregate/deepface_simpson_sensitivity.csv",
        "data/human-unesp-aggregate/face_category_composition.csv",
        "data/human-unesp-aggregate/image_face_count_summary.csv",
        "data/human-unesp-aggregate/human_validation_metadata.json",
    }
    assert {
        path
        for path in checkpoint_module.PUBLIC_ARTIFACTS
        if path.startswith("output/validation/vitor-64-")
    } == {
        "output/validation/vitor-64-detectors/detector_disagreement_summary.csv",
        "output/validation/vitor-64-detectors/detector_disagreement_bootstrap_ci.csv",
        "output/validation/vitor-64-detectors/detector_validation_metadata.json",
        "output/validation/vitor-64-fairface/deepface_fairface_comparison.csv",
        "output/validation/vitor-64-fairface/deepface_fairface_concordance.csv",
        "output/validation/vitor-64-fairface/confusion_race.csv",
        "output/validation/vitor-64-fairface/confusion_gender.csv",
        "output/validation/vitor-64-fairface/fairface_run_metadata.json",
    }


def test_artifact_record_hashes_file(tmp_path: Path) -> None:
    artifact = tmp_path / "public.csv"
    artifact.write_bytes(b"a,b\n1,2\n")

    record = checkpoint_module.artifact_record(tmp_path, "public.csv")

    assert record == {
        "path": "public.csv",
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "bytes": artifact.stat().st_size,
    }


@pytest.mark.parametrize("path", checkpoint_module.EXCLUDED_PRIVATE_PREFIXES)
def test_artifact_record_rejects_private_prefixes(tmp_path: Path, path: str) -> None:
    with pytest.raises(ValueError, match="Private artifact"):
        checkpoint_module.artifact_record(tmp_path, f"{path}responses.csv")
