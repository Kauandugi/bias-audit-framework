"""Valida os artefatos produzidos pela execução completa do notebook oficial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import mannwhitneyu, wilcoxon


IMAGE_COLUMNS = {
    "imagem_id", "caminho_relativo", "nome_arquivo", "prompt_id",
    "tipo_prompt", "modelo_ia", "num_faces", "status_deteccao",
    "erro_deteccao", "clip_latin_american", "clip_not_latin_american",
    "clip_margin",
}
FACE_COLUMNS = {
    "imagem_id", "face_id", "num_faces", "bbox_x", "bbox_y", "bbox_w",
    "bbox_h", "raca", "confianca_raca", "genero", "confianca_genero",
}
EXPECTED_FILES = {
    "audit_images.csv", "audit_faces.csv", "run_metadata.json",
    "mann_whitney_overall.csv", "wilcoxon_sensitivity.csv",
    "mann_whitney_by_model.csv", "grafico_raca.png", "grafico_genero.png",
    "grafico_clip_latino.png",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def normalize_prompt_group(value: object) -> str:
    text = str(value).casefold()
    if "inclus" in text:
        return "inclusive"
    if "neutral" in text or "neutro" in text:
        return "neutral"
    return text.strip()


def validate_outputs(output_dir: Path, expected_images: int = 64) -> dict[str, int | float]:
    missing_files = EXPECTED_FILES - {path.name for path in output_dir.iterdir() if path.is_file()}
    require(not missing_files, f"Arquivos ausentes: {sorted(missing_files)}")

    images = pd.read_csv(output_dir / "audit_images.csv")
    faces = pd.read_csv(output_dir / "audit_faces.csv")
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))

    require(set(images.columns) == IMAGE_COLUMNS, "Schema de audit_images.csv divergente")
    require(set(faces.columns) == FACE_COLUMNS, "Schema de audit_faces.csv divergente")
    require(len(images) == expected_images, f"Esperadas {expected_images} imagens; obtidas {len(images)}")
    require(images["imagem_id"].notna().all() and images["imagem_id"].is_unique, "imagem_id inválido")
    require(images["caminho_relativo"].is_unique, "Caminho relativo duplicado")
    require(faces["face_id"].notna().all() and faces["face_id"].is_unique, "face_id inválido")
    require(set(faces["imagem_id"]).issubset(set(images["imagem_id"])), "Rosto sem imagem correspondente")

    expected_counts = images.set_index("imagem_id")["num_faces"]
    observed_counts = faces.groupby("imagem_id").size().reindex(expected_counts.index, fill_value=0)
    require(observed_counts.equals(expected_counts), "num_faces diverge das linhas faciais")
    require(not any(column.startswith("clip_") for column in faces.columns), "CLIP presente na tabela facial")

    require((faces[["bbox_w", "bbox_h"]] > 0).all().all(), "Caixa facial com dimensão não positiva")
    require((faces[["bbox_x", "bbox_y"]] >= 0).all().all(), "Caixa facial com origem negativa")
    require(faces["confianca_raca"].between(0, 100).all(), "Confiança racial fora de [0, 100]")
    require(faces["confianca_genero"].between(0, 100).all(), "Confiança de gênero fora de [0, 100]")

    require(images["clip_latin_american"].between(-1, 1).all(), "Cosseno latino fora de [-1, 1]")
    require(images["clip_not_latin_american"].between(-1, 1).all(), "Cosseno não latino fora de [-1, 1]")
    require(images["clip_margin"].between(-2, 2).all(), "Margem CLIP fora de [-2, 2]")
    reconstructed = images["clip_latin_american"] - images["clip_not_latin_american"]
    require(np.allclose(images["clip_margin"], reconstructed, atol=1e-12), "Margem CLIP inconsistente")

    data = images.copy()
    data["grupo"] = data["tipo_prompt"].map(normalize_prompt_group)
    inclusive = data.loc[data["grupo"] == "inclusive", "clip_margin"]
    neutral = data.loc[data["grupo"] == "neutral", "clip_margin"]
    u_statistic, u_p_value = mannwhitneyu(inclusive, neutral, alternative="two-sided")
    exported_u = pd.read_csv(output_dir / "mann_whitney_overall.csv").iloc[0]
    require(np.isclose(exported_u["u_statistic"], u_statistic), "U exportado divergente")
    require(np.isclose(exported_u["p_value"], u_p_value), "p-value de Mann-Whitney divergente")

    pairs = data.pivot_table(
        index=["modelo_ia", "prompt_id"], columns="grupo", values="clip_margin", aggfunc="first"
    ).dropna(subset=["inclusive", "neutral"])
    w_statistic, w_p_value = wilcoxon(pairs["inclusive"] - pairs["neutral"], alternative="two-sided")
    exported_w = pd.read_csv(output_dir / "wilcoxon_sensitivity.csv").iloc[0]
    require(np.isclose(exported_w["w_statistic"], w_statistic), "W exportado divergente")
    require(np.isclose(exported_w["p_value"], w_p_value), "p-value de Wilcoxon divergente")

    require(metadata.get("cuda_available") is True, "Execução não registrou CUDA")
    require(metadata.get("gpu_name") == "Tesla T4", "GPU registrada não é Tesla T4")
    require(metadata.get("clip_scoring") == "L2-normalized cosine; contrastive margin", "CLIP não registrado como cosseno L2")

    for path in output_dir.glob("*.png"):
        with Image.open(path) as image:
            image.verify()

    return {
        "images": len(images),
        "faces": len(faces),
        "mann_whitney_p": float(u_p_value),
        "wilcoxon_p": float(w_p_value),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path("execucao") / "BiasAuditFW outputs",
    )
    args = parser.parse_args()
    summary = validate_outputs(args.output_dir)
    print("Validação concluída:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
