"""Select a blinded human-validation sample and create counterbalanced orders."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from biasauditfw.sampling import select_human_validation_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-csv", type=Path, nargs=2, required=True)
    parser.add_argument("--judge-pointwise", type=Path, nargs=2, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images = pd.concat(
        [pd.read_csv(path) for path in args.images_csv], ignore_index=True
    )
    pointwise = pd.concat(
        [pd.read_csv(path) for path in args.judge_pointwise], ignore_index=True
    )
    sample = select_human_validation_sample(images, pointwise, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output_dir / "human_validation_sample_internal.csv", index=False)

    rng = np.random.default_rng(args.seed)
    order_a = sample[["imagem_id", "caminho_relativo"]].sample(
        frac=1, random_state=args.seed
    ).reset_index(drop=True)
    order_a.insert(0, "form_position", range(1, len(order_a) + 1))
    order_b = order_a.iloc[::-1].reset_index(drop=True)
    # Rotate the reversed order to avoid placing the same images at both extremes.
    offset = int(rng.integers(1, len(order_b)))
    order_b = pd.concat([order_b.iloc[offset:], order_b.iloc[:offset]], ignore_index=True)
    order_b["form_position"] = range(1, len(order_b) + 1)
    order_a.to_csv(args.output_dir / "human_form_order_a.csv", index=False)
    order_b.to_csv(args.output_dir / "human_form_order_b.csv", index=False)
    print("Created 24-image internal sample and two blinded form orders.")


if __name__ == "__main__":
    main()
