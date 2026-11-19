"""Export blinded image assets for the four UNESP form versions."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image


def _dataset_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        dataset_id, separator, raw_path = value.partition("=")
        if not separator or not dataset_id.strip() or not raw_path.strip():
            raise ValueError(
                "Dataset roots must use DATASET_ID=PATH, for example original-64=./images"
            )
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        roots[dataset_id.strip()] = path
    return roots


def export_form_assets(
    panel_csvs: list[Path], dataset_roots: dict[str, Path], output_dir: Path
) -> pd.DataFrame:
    """Copy images under neutral names and return a non-identifying public index."""

    prepared_panels: list[tuple[str, pd.DataFrame, list[Path]]] = []
    seen_codes: set[str] = set()
    for panel_csv in panel_csvs:
        panel = pd.read_csv(panel_csv)
        required = {"form_code", "form_position", "dataset_id", "caminho_relativo"}
        missing = required - set(panel)
        if missing:
            raise ValueError(f"Panel {panel_csv} is missing columns: {sorted(missing)}")
        codes = panel["form_code"].dropna().astype(str).unique()
        if len(codes) != 1:
            raise ValueError(f"Panel {panel_csv} must contain exactly one form_code")
        form_code = codes[0].upper()
        if form_code in seen_codes:
            raise ValueError(f"Duplicate form_code: {form_code}")
        seen_codes.add(form_code)
        expected_positions = list(range(1, len(panel) + 1))
        positions = panel["form_position"].astype(int).tolist()
        if positions != expected_positions:
            raise ValueError(f"Panel {form_code} positions must be consecutive from one")
        sources: list[Path] = []
        for row in panel.itertuples(index=False):
            dataset_id = str(row.dataset_id)
            if dataset_id not in dataset_roots:
                raise KeyError(f"No root configured for dataset {dataset_id!r}")
            root = dataset_roots[dataset_id]
            source = (root / str(row.caminho_relativo)).resolve()
            if not source.is_file() or root not in source.parents:
                raise FileNotFoundError(source)
            with Image.open(source) as image:
                image.verify()
            sources.append(source)
        prepared_panels.append((form_code, panel, sources))

    public_rows: list[dict[str, object]] = []
    for form_code, panel, sources in prepared_panels:
        form_dir = output_dir / form_code
        form_dir.mkdir(parents=True, exist_ok=True)
        for row, source in zip(panel.itertuples(index=False), sources, strict=True):
            with Image.open(source) as image:
                extension = (image.format or source.suffix.removeprefix(".")).lower()
            if extension == "jpeg":
                extension = "jpg"
            position = int(row.form_position)
            target = form_dir / f"imagem_{position:02d}.{extension}"
            shutil.copy2(source, target)
            public_rows.append(
                {
                    "form_code": form_code,
                    "form_position": position,
                    "image_label": f"Imagem {position} de {len(panel)}",
                    "asset_path": target.relative_to(output_dir).as_posix(),
                }
            )

    public_index = pd.DataFrame(public_rows).sort_values(
        ["form_code", "form_position"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    public_index.to_csv(output_dir / "public_sequence.csv", index=False)
    return public_index.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-csv", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--dataset-root",
        action="append",
        required=True,
        help="Repeat DATASET_ID=PATH for every corpus.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    public_index = export_form_assets(
        args.panel_csv,
        _dataset_roots(args.dataset_root),
        args.output_dir,
    )
    counts = public_index.groupby("form_code").size().to_dict()
    print(f"Exported blinded form assets: {counts}")


if __name__ == "__main__":
    main()
