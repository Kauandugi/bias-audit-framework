from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).parents[1]
HUMAN_DIR = ROOT / "data" / "human-unesp-aggregate"
TCC_DIR = ROOT / "output" / "tcc-final-prism-v3"


def _row(frame: pd.DataFrame, **filters: object) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        selected = selected.loc[selected[column].eq(value)]
    assert len(selected) == 1
    return selected.iloc[0]


def test_human_results_match_public_narrative() -> None:
    metadata = json.loads(
        (HUMAN_DIR / "human_validation_metadata.json").read_text(encoding="utf-8")
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert metadata["responses_total"] == 72
    assert metadata["responses_consented"] == 70
    assert metadata["responses_declined"] == 2
    assert metadata["responses_excluded"] == 2
    assert metadata["participants"] == 70
    assert metadata["images"] == 48
    assert metadata["private_rows"] == 3360
    assert metadata["collection_preceded_final_ethics_opinion"] is True
    assert metadata["simpson_estimator"] == "unbiased_gini_simpson"

    assert "70 participantes" in readme
    assert "3.360 avaliações de imagens por participantes" in readme
    assert "16.800 notas" in readme

    # The final narrative must use the small-sample-corrected facial index.
    paired = pd.read_csv(HUMAN_DIR / "deepface_simpson_paired.csv")
    original_racial = _row(paired, dataset_id="original-64", dimension="racial")
    replication_racial = _row(
        paired, dataset_id="vitor-64", dimension="racial"
    )
    assert original_racial["n_pairs"] == 31
    assert original_racial["median_paired_difference"] == pytest.approx(
        0.143883, abs=1e-6
    )
    assert original_racial["p_value_holm"] == pytest.approx(0.013916, abs=1e-6)
    assert replication_racial["n_pairs"] == 30
    assert replication_racial["p_value_holm"] == pytest.approx(0.044908, abs=1e-6)

    sensitivity = pd.read_csv(HUMAN_DIR / "deepface_simpson_sensitivity.csv")
    replication_racial_without_p8 = _row(
        sensitivity,
        dataset_id="vitor-64",
        dimension="racial",
        analysis_variant="exclude_P8",
    )
    assert replication_racial_without_p8["p_value_holm"] == pytest.approx(
        0.194495, abs=1e-6
    )

    participant = pd.read_csv(HUMAN_DIR / "human_participant_sensitivity.csv")
    original_racial_participant = _row(
        participant,
        human_corpus_id="original-64",
        dimension="racial",
    )
    assert original_racial_participant["n_nonzero_participants"] == 67
    assert original_racial_participant[
        "median_participant_mean_difference"
    ] == pytest.approx(0.541667, abs=1e-6)

    convergence = pd.read_csv(HUMAN_DIR / "human_framework_convergence.csv")
    replication_racial_convergence = _row(
        convergence,
        dataset_id="vitor-64",
        dimension="racial",
        automated_metric="deepface_racial_simpson",
    )
    assert replication_racial_convergence["spearman_p_value_holm"] == pytest.approx(
        0.1229, abs=1e-4
    )
    assert replication_racial_convergence[
        "partial_spearman_p_value_face_count_holm"
    ] == pytest.approx(0.033762, abs=1e-6)


def test_tcc_narrative_matches_public_results_when_available() -> None:
    required = (
        TCC_DIR / "cap-metodologia.tex",
        TCC_DIR / "cap-resultados.tex",
        TCC_DIR / "cap-conc.tex",
        TCC_DIR / "main.tex",
    )
    if not all(path.is_file() for path in required):
        pytest.skip("The public code package does not include the academic LaTeX source")

    methodology, results, conclusion, main = (
        path.read_text(encoding="utf-8") for path in required
    )
    assert "Setenta pessoas" in methodology
    assert "70 participantes" in main
    assert "3.360 avaliações participante--imagem" in main
    assert "16.800 notas" in main
    assert "12 pares" in main
    assert "A sensibilidade por participante" in conclusion
    assert "Parecer Consubstanciado nº 8.854.629" in methodology
    assert "UNESP" in methodology
    assert "0,144 [0,013; 0,294]" in results
    assert "0,0139" in results
    assert "deixaram de ser significativos" in results
    assert "resultado parcial é tratado como evidência exploratória" in results
