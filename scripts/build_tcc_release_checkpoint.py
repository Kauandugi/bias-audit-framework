"""Build a hash manifest for the public artifacts used by the TCC."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


PUBLIC_ARTIFACTS = (
    "Framework_Auditoria_Viés_IA_Generativa.ipynb",
    "output/TCC_BiasAuditFW_Pos_Banca.pdf",
    "data/schema2/audit_images.csv",
    "data/schema2/audit_faces.csv",
    "data/schema2/run_metadata.json",
    "data/schema2/wilcoxon_primary.csv",
    "data/schema2/mann_whitney_sensitivity.csv",
    "data/schema2-vitor/audit_images.csv",
    "data/schema2-vitor/audit_faces.csv",
    "data/schema2-vitor/run_metadata.json",
    "data/schema2-vitor/wilcoxon_primary.csv",
    "data/schema2-vitor/mann_whitney_sensitivity.csv",
    "output/validation/vitor-64-detectors/detector_disagreement_summary.csv",
    "output/validation/vitor-64-detectors/detector_disagreement_bootstrap_ci.csv",
    "output/validation/vitor-64-detectors/detector_validation_metadata.json",
    "output/validation/vitor-64-fairface/deepface_fairface_comparison.csv",
    "output/validation/vitor-64-fairface/deepface_fairface_concordance.csv",
    "output/validation/vitor-64-fairface/confusion_race.csv",
    "output/validation/vitor-64-fairface/confusion_gender.csv",
    "output/validation/vitor-64-fairface/fairface_run_metadata.json",
    "output/validation/wider-face/wider_detector_summary.csv",
    "output/validation/wider-face/wider_detector_bootstrap_ci.csv",
    "output/validation/wider-face/detector_validation_metadata.json",
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
)

EXCLUDED_PRIVATE_PREFIXES = (
    "data/private/",
    "Resultados Formulario/",
    "Parecer Comite etica/",
    "output/validation/human-unesp/",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(repo_root: Path, relative_path: str) -> dict[str, object]:
    normalized = relative_path.replace("\\", "/")
    if normalized.startswith(EXCLUDED_PRIVATE_PREFIXES):
        raise ValueError(f"Private artifact cannot enter checkpoint: {normalized}")
    path = repo_root / Path(normalized)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": normalized,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo_root, text=True, encoding="utf-8"
    ).strip()


def git_snapshot(repo_root: Path) -> dict[str, object]:
    status_lines = [
        line for line in _git(repo_root, "status", "--porcelain=v1").splitlines()
        if line
    ]
    return {
        "head": _git(repo_root, "rev-parse", "HEAD"),
        "branch": _git(repo_root, "branch", "--show-current"),
        "dirty": bool(status_lines),
        "dirty_entry_count": len(status_lines),
    }


def build_checkpoint(repo_root: Path) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "repository": git_snapshot(repo_root),
        "model_pins": {
            "clip_model_id": "openai/clip-vit-base-patch32",
            "clip_model_revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        },
        "artifacts": [
            artifact_record(repo_root, relative_path)
            for relative_path in PUBLIC_ARTIFACTS
        ],
        "caveats": [
            "Frozen canonical outputs predate the CLIP revision field and were not rewritten.",
            "Individual human responses, profiles, consent records, and the private image map are excluded.",
            "Only authorized, non-identifying human aggregates are included.",
            "MLLM pilot artifacts are diagnostic and are not evidence used in the conclusions.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/tcc-release-checkpoint.json"),
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Write a candidate checkpoint even when the repository is dirty.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    checkpoint = build_checkpoint(repo_root)
    repository = checkpoint["repository"]
    if repository["dirty"] and not args.allow_dirty:
        raise RuntimeError(
            "Repository is dirty. Review changes and commit before building the final checkpoint, "
            "or use --allow-dirty for a non-final candidate."
        )
    output = args.output
    if not output.is_absolute():
        output = repo_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(checkpoint, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": output.relative_to(repo_root).as_posix(),
                "artifacts": len(checkpoint["artifacts"]),
                "dirty": repository["dirty"],
            }
        )
    )


if __name__ == "__main__":
    main()
