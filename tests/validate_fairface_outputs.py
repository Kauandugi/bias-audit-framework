"""Independently validate frozen DeepFace/FairFace comparison artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from biasauditfw.validation import concordance_report


def validate(
    output_dir: Path,
    *,
    expected_images: int,
    expected_faces: int,
) -> dict[str, object]:
    required = {
        "deepface_fairface_comparison.csv",
        "deepface_fairface_concordance.csv",
        "confusion_race.csv",
        "confusion_gender.csv",
        "fairface_run_metadata.json",
    }
    missing = sorted(name for name in required if not (output_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"Missing FairFace validation artifacts: {missing}")

    comparison = pd.read_csv(output_dir / "deepface_fairface_comparison.csv")
    saved_report = pd.read_csv(
        output_dir / "deepface_fairface_concordance.csv"
    ).sort_values("attribute", ignore_index=True)
    metadata = json.loads(
        (output_dir / "fairface_run_metadata.json").read_text(encoding="utf-8")
    )

    required_columns = {
        "imagem_id",
        "face_id",
        "deepface_raca",
        "deepface_genero",
        "fairface_raca",
        "fairface_genero",
    }
    if missing_columns := required_columns - set(comparison):
        raise AssertionError(
            f"Comparison is missing columns: {sorted(missing_columns)}"
        )
    if comparison["face_id"].duplicated().any():
        raise AssertionError("Duplicate face_id values in FairFace comparison")
    if len(comparison) != expected_faces:
        raise AssertionError(
            f"Expected {expected_faces} face comparisons, found {len(comparison)}"
        )
    if comparison["imagem_id"].nunique() != expected_images:
        raise AssertionError(
            f"Expected {expected_images} represented images, found "
            f"{comparison['imagem_id'].nunique()}"
        )

    recalculated_report, recalculated_matrices = concordance_report(comparison)
    recalculated_report = recalculated_report.sort_values(
        "attribute", ignore_index=True
    )
    pd.testing.assert_frame_equal(
        saved_report,
        recalculated_report,
        check_dtype=False,
        atol=1e-12,
        rtol=1e-12,
    )
    for attribute, recalculated in recalculated_matrices.items():
        saved = pd.read_csv(
            output_dir / f"confusion_{attribute}.csv", index_col=0
        )
        saved.index.name = recalculated.index.name
        pd.testing.assert_frame_equal(
            saved,
            recalculated,
            check_dtype=False,
            check_names=True,
        )

    expected_metadata = {
        "images": expected_images,
        "faces": expected_faces,
        "comparison_rows": expected_faces,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise AssertionError(
                f"Metadata {key}={metadata.get(key)!r}, expected {expected!r}"
            )
    interpretation = str(metadata.get("interpretation", "")).casefold()
    if "pseudo-reference" not in interpretation or "not demographic ground truth" not in interpretation:
        raise AssertionError("Metadata must state the pseudo-reference limitation")

    return {
        "images": expected_images,
        "faces": expected_faces,
        "attributes": recalculated_report.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-images", type=int, default=64)
    parser.add_argument("--expected-faces", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            validate(
                args.output_dir,
                expected_images=args.expected_images,
                expected_faces=args.expected_faces,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
