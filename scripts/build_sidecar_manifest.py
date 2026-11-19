"""Create a schema 2.0 manifest from adjacent metadata.json files."""

from __future__ import annotations

import argparse
from pathlib import Path

from biasauditfw.manifest import build_sidecar_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--expected-images", type=int)
    args = parser.parse_args()

    manifest = build_sidecar_manifest(
        args.input_root,
        dataset_id=args.dataset_id,
        expected_images=args.expected_images,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(f"Wrote {len(manifest)} rows to {args.output}")


if __name__ == "__main__":
    main()
