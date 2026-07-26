from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from biasauditfw import DatasetConfig, discover_dataset, export_results, process_dataset
from biasauditfw.reporting import export_statistical_reports
from conftest import make_image


def test_arbitrary_directory_to_validated_schema2_outputs(tmp_path: Path) -> None:
    first = make_image(tmp_path / "deep" / "one" / "neutral.PNG")
    second = make_image(tmp_path / "another place" / "inclusive.webp", (1, 2, 3))
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "relative_path": first.relative_to(tmp_path).as_posix(),
                "source_image_id": "N",
                "modelo_ia": "Mock",
                "tipo_prompt": "Neutral",
                "prompt_id": "P1",
                "pair_id": "Mock_P1",
                "analysis_eligible": True,
            },
            {
                "relative_path": second.relative_to(tmp_path).as_posix(),
                "source_image_id": "I",
                "modelo_ia": "Mock",
                "tipo_prompt": "Inclusive",
                "prompt_id": "P1",
                "pair_id": "Mock_P1",
                "analysis_eligible": True,
            },
        ]
    ).to_csv(manifest, index=False)
    config = DatasetConfig(
        input_root=tmp_path,
        output_dir=tmp_path / "outputs_v2",
        dataset_id="integration",
        manifest_path=manifest,
        expected_images=2,
    )
    discovery = discover_dataset(config)

    def faces(_: Path):
        return (
            [
                {
                    "bbox_x": 1,
                    "bbox_y": 1,
                    "bbox_w": 5,
                    "bbox_h": 5,
                    "raca": "white",
                    "confianca_raca": 80.0,
                    "genero": "Man",
                    "confianca_genero": 80.0,
                }
            ],
            "detected",
            None,
        )

    def clip(path: Path):
        shift = 0.2 if "inclusive" in path.name else 0.0
        return {
            "clip_racial_diversity_cosine": 0.4 + shift,
            "clip_racial_homogeneity_cosine": 0.2,
            "clip_racial_diversity_margin": 0.2 + shift,
            "clip_gender_diversity_cosine": 0.3 + shift,
            "clip_gender_homogeneity_cosine": 0.1,
            "clip_gender_diversity_margin": 0.2 + shift,
        }

    images, face_rows = process_dataset(discovery.images, faces, clip)
    metadata = {
        "clip_scoring": "L2-normalized cosine prototype margins",
        "cuda_available": False,
        "gpu_name": None,
    }
    export_results(
        images,
        face_rows,
        discovery.report,
        config.output_dir,
        metadata,
        expected_images=2,
    )
    export_statistical_reports(images, face_rows, config.output_dir)

    validator = Path(__file__).parent / "validate_full_outputs.py"
    result = subprocess.run(
        [
            sys.executable,
            str(validator),
            str(config.output_dir),
            "--expected-images",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert '"images": 2' in result.stdout

