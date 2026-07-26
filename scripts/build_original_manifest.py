"""Build the explicit metadata manifest for the original 64-image corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODEL_CODES = {
    "DALL-E": "DALLE",
    "Midjourney": "MJD",
    "Gemini (NanoBanana)": "NB",
    "Whisk": "WHISK",
}


def build_manifest(legacy_images: Path) -> pd.DataFrame:
    source = pd.read_csv(legacy_images)
    required = {
        "caminho_relativo",
        "modelo_ia",
        "tipo_prompt",
        "prompt_id",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Legacy table is missing columns: {sorted(missing)}")

    records = []
    for row in source.itertuples(index=False):
        prompt_group = str(row.tipo_prompt).casefold()
        suffix = "I" if "inclusive" in prompt_group else "N"
        model_code = MODEL_CODES[str(row.modelo_ia)]
        prompt_id = str(row.prompt_id).upper()
        records.append(
            {
                "relative_path": str(row.caminho_relativo).replace("\\", "/"),
                "source_image_id": f"{model_code}_{prompt_id}_{suffix}",
                "modelo_ia": row.modelo_ia,
                "tipo_prompt": row.tipo_prompt,
                "prompt_id": prompt_id,
                "pair_id": f"{model_code}_{prompt_id}",
                "analysis_eligible": True,
            }
        )

    manifest = pd.DataFrame(records).sort_values(
        ["modelo_ia", "prompt_id", "tipo_prompt"]
    )
    if len(manifest) != 64:
        raise ValueError(f"Expected 64 rows, found {len(manifest)}.")
    if not manifest["relative_path"].is_unique:
        raise ValueError("relative_path must be unique.")
    if not manifest["source_image_id"].is_unique:
        raise ValueError("source_image_id must be unique.")
    pair_sizes = manifest.groupby("pair_id").size()
    if not pair_sizes.eq(2).all():
        raise ValueError("Each pair_id must identify one neutral/inclusive pair.")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_images", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = build_manifest(args.legacy_images)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(f"Wrote {len(manifest)} rows to {args.output}")


if __name__ == "__main__":
    main()
