"""Recover the private form assets and bind positions to content hashes."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

import pandas as pd
from PIL import Image

from biasauditfw.ingestion import make_image_id


MODEL_ALIASES = {
    (2025, "DALL-E"): "DALL-E",
    (2025, "Nano Banana"): "Gemini (NanoBanana)",
    (2025, "Midjourney"): "Midjourney",
    (2025, "Whisk"): "Whisk",
    (2026, "DALL-E"): "OpenAI image_gen via Codex",
    (2026, "Nano Banana"): "Gemini",
    (2026, "Midjourney"): "Midjourney",
    (2026, "Whisk"): "Whisk",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_drive_file(file_id: str, target_stem: Path) -> Path:
    temporary = target_stem.with_suffix(".download")
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        urllib.request.urlretrieve(url, temporary)
        with Image.open(temporary) as image:
            image.verify()
            extension = (image.format or "bin").casefold()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if extension == "jpeg":
        extension = "jpg"
    target = target_stem.with_suffix(f".{extension}")
    temporary.replace(target)
    return target


def _prompt_group(value: object) -> str:
    return "inclusive" if str(value).casefold().startswith("inclus") else "neutral"


def _canonical_metadata_match(
    canonical: pd.DataFrame, source: object
) -> pd.DataFrame:
    year = int(source.ano)
    dataset_id = "original-64" if year == 2025 else "vitor-64"
    model = MODEL_ALIASES[(year, str(source.produto))]
    prompt = str(source.prompt).upper()
    groups = canonical["tipo_prompt"].map(_prompt_group)
    return canonical.loc[
        canonical["dataset_id"].eq(dataset_id)
        & canonical["modelo_ia"].eq(model)
        & canonical["prompt_id"].astype(str).str.upper().eq(prompt)
        & groups.eq(_prompt_group(source.variante))
    ]


def build_private_map(
    workbook: Path,
    canonical_paths: list[Path],
    assets_dir: Path,
) -> pd.DataFrame:
    source_map = pd.read_excel(workbook, sheet_name="mapeamento", engine="openpyxl")
    required = {"posicao", "prompt", "variante", "produto", "ano", "file_id"}
    if missing := required - set(source_map):
        raise ValueError(f"Mapping sheet is missing columns: {sorted(missing)}")
    canonical = pd.concat(
        [pd.read_csv(path) for path in canonical_paths], ignore_index=True
    )
    if canonical["content_sha256"].duplicated().any():
        raise ValueError("Canonical image tables contain duplicate hashes")
    by_hash = canonical.set_index("content_sha256", drop=False)

    assets_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for source in source_map.sort_values("posicao").itertuples(index=False):
        position = int(source.posicao)
        existing = [
            *assets_dir.glob(f"form_position_{position:02d}.*"),
            *assets_dir.glob(f"position-{position:02d}.*"),
        ]
        existing = [path for path in existing if path.suffix != ".download"]
        asset: Path | None = existing[0] if existing else None
        candidates = _canonical_metadata_match(canonical, source)
        if asset is None and len(candidates) != 1:
            try:
                asset = _download_drive_file(
                    str(source.file_id), assets_dir / f"form_position_{position:02d}"
                )
            except Exception:
                asset = None
        digest = _sha256(asset) if asset is not None else None
        if digest is not None and digest in by_hash.index:
            canonical_row = by_hash.loc[digest]
            row = canonical_row.to_dict()
            row["canonical_match"] = True
            row["framework_metrics_available"] = True
            verification_method = "downloaded_content_sha256"
        elif len(candidates) == 1:
            canonical_row = candidates.iloc[0]
            row = canonical_row.to_dict()
            digest = str(canonical_row.content_sha256)
            row["canonical_match"] = True
            row["framework_metrics_available"] = True
            verification_method = "unique_frozen_manifest_metadata"
        elif digest is not None:
            dataset_id = "vitor-form-supplement"
            model = MODEL_ALIASES[(int(source.ano), str(source.produto))]
            prompt_id = str(source.prompt).upper()
            variant = str(source.variante).casefold()
            prompt_type = (
                "Inclusive Prompts" if variant.startswith("inclus") else "Neutral Prompts"
            )
            row = {
                "dataset_id": dataset_id,
                "imagem_id": make_image_id(dataset_id, digest),
                "content_sha256": digest,
                "caminho_relativo": asset.name,
                "nome_arquivo": asset.name,
                "source_image_id": f"FORM-{position:02d}",
                "prompt_id": prompt_id,
                "pair_id": f"{dataset_id}:{model}:{prompt_id}",
                "tipo_prompt": prompt_type,
                "modelo_ia": model,
                "analysis_eligible": False,
                "canonical_match": False,
                "framework_metrics_available": False,
            }
            verification_method = "downloaded_supplement_sha256"
        else:
            raise ValueError(
                f"Position {position} could not be downloaded or matched uniquely"
            )
        row.update(
            {
                "form_code": "UNESP-2026",
                "form_position": position,
                "form_file_id": str(source.file_id),
                "form_asset_sha256": digest,
                "form_asset_path": asset.as_posix() if asset is not None else pd.NA,
                "verification_method": verification_method,
                "human_corpus_id": (
                    "original-64" if int(source.ano) == 2025 else "vitor-form-24"
                ),
                "human_pair_id": (
                    f"{int(source.ano)}:{MODEL_ALIASES[(int(source.ano), str(source.produto))]}:"
                    f"{str(source.prompt).upper()}"
                ),
                "human_prompt_group": _prompt_group(source.variante),
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("form_position").reset_index(drop=True)
    if result["form_position"].tolist() != list(range(1, len(result) + 1)):
        raise ValueError("Form positions must be consecutive from one")
    if result["imagem_id"].duplicated().any():
        raise ValueError("A form image was repeated")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--canonical-images", type=Path, nargs="+", required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--output-map", type=Path, required=True)
    args = parser.parse_args()

    mapping = build_private_map(
        args.workbook,
        args.canonical_images,
        args.assets_dir,
    )
    args.output_map.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(args.output_map, index=False)
    matched = int(mapping["canonical_match"].sum())
    print(
        f"Mapped {len(mapping)} form positions by content hash; "
        f"{matched} match canonical framework images and "
        f"{len(mapping) - matched} are private form supplements."
    )


if __name__ == "__main__":
    main()
