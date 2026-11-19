"""Create the explicit schema 2.0 manifest for Vitor's replication corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from biasauditfw.manifest import build_path_manifest


MODEL_NAMES = {
    "ChatGPT": "OpenAI image_gen via Codex",
    "Gemini": "Gemini",
    "Midjourney": "Midjourney",
    "Whisk": "Whisk",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset-id", default="vitor-replication-2026")
    parser.add_argument("--expected-images", type=int)
    args = parser.parse_args()

    manifest = build_path_manifest(
        args.input_root,
        dataset_id=args.dataset_id,
        model_names=MODEL_NAMES,
        expected_images=args.expected_images,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    incomplete = manifest.loc[~manifest["analysis_eligible"], "relative_path"]
    print(f"Wrote {len(manifest)} rows to {args.output}")
    print(f"Inference-eligible images: {int(manifest['analysis_eligible'].sum())}")
    if not incomplete.empty:
        print("Incomplete-pair images excluded from inference:")
        for path in incomplete:
            print(f"- {path}")


if __name__ == "__main__":
    main()
