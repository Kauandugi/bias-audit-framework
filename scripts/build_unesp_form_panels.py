"""Build four private, blinded UNESP form orders from two frozen corpora."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from biasauditfw.sampling import build_unesp_form_orders


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-csv", type=Path, nargs=2, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    images = pd.concat(
        [pd.read_csv(path) for path in args.images_csv], ignore_index=True
    )
    orders = build_unesp_form_orders(images, seed=args.seed, allow_unpaired=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for form_code, order in orders.items():
        order.to_csv(
            args.output_dir / f"unesp_form_{form_code.lower()}_private.csv",
            index=False,
        )
    sizes = {code: len(order) for code, order in orders.items()}
    print(f"Created private orders with sizes: {sizes}")


if __name__ == "__main__":
    main()
