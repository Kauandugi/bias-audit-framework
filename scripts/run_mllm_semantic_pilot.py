"""Run a detector-screened local-judge pilot before a full MLLM batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from biasauditfw.judges import (
    GeminiProvider,
    GroqProvider,
    JudgeRunStore,
    LocalGemmaProvider,
    LocalQwen3VLProvider,
    LocalQwenProvider,
    build_pair_tasks,
    evaluate_semantic_gate,
    export_judge_tables,
    run_pairwise,
    run_pointwise,
    select_semantic_pilot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-csv", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-count", type=int, default=4)
    parser.add_argument("--min-faces-per-image", type=int, default=2)
    parser.add_argument(
        "--provider",
        choices=("gemini", "groq", "qwen", "qwen3", "gemma"),
        default="qwen3",
    )
    parser.add_argument("--gemini-model-id", default="gemini-3.6-flash")
    parser.add_argument("--groq-model-id", default="qwen/qwen3.8-27b")
    parser.add_argument(
        "--qwen-model-id", default="Qwen/Qwen2.5-VL-7B-Instruct"
    )
    parser.add_argument("--qwen-revision", default="main")
    parser.add_argument(
        "--qwen-quantization", choices=("none", "4bit-nf4"), default="4bit-nf4"
    )
    parser.add_argument("--qwen-min-visual-tokens", type=int, default=256)
    parser.add_argument("--qwen-max-visual-tokens", type=int, default=768)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=128)
    parser.add_argument("--qwen3-model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--qwen3-revision", default="main")
    parser.add_argument(
        "--qwen3-quantization", choices=("none", "4bit-nf4"), default="4bit-nf4"
    )
    parser.add_argument("--qwen3-max-new-tokens", type=int, default=128)
    parser.add_argument("--gemma-model-id", default="google/gemma-3-4b-it")
    parser.add_argument("--gemma-revision", default="main")
    parser.add_argument(
        "--gemma-quantization", choices=("none", "4bit-nf4"), default="4bit-nf4"
    )
    parser.add_argument("--gemma-max-new-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = pd.read_csv(args.images_csv)
    selected = select_semantic_pilot(
        images,
        pair_count=args.pair_count,
        min_faces_per_image=args.min_faces_per_image,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "semantic_pilot_sample.csv"
    selected.to_csv(sample_path, index=False)

    if args.provider == "gemini":
        provider = GeminiProvider(model_id=args.gemini_model_id)
    elif args.provider == "groq":
        provider = GroqProvider(model_id=args.groq_model_id)
    elif args.provider == "qwen":
        provider = LocalQwenProvider(
            model_id=args.qwen_model_id,
            model_revision=args.qwen_revision,
            min_pixels=args.qwen_min_visual_tokens * 28 * 28,
            max_pixels=args.qwen_max_visual_tokens * 28 * 28,
            max_new_tokens=args.qwen_max_new_tokens,
            quantization=args.qwen_quantization,
        )
    elif args.provider == "qwen3":
        provider = LocalQwen3VLProvider(
            model_id=args.qwen3_model_id,
            model_revision=args.qwen3_revision,
            max_new_tokens=args.qwen3_max_new_tokens,
            quantization=args.qwen3_quantization,
        )
    else:
        provider = LocalGemmaProvider(
            model_id=args.gemma_model_id,
            model_revision=args.gemma_revision,
            max_new_tokens=args.gemma_max_new_tokens,
            quantization=args.gemma_quantization,
        )
    store = JudgeRunStore(args.output_dir / "judge_runs.jsonl")
    run_pointwise([provider], selected, args.input_root, store)
    run_pairwise(
        [provider],
        build_pair_tasks(selected, args.input_root),
        store,
    )
    pointwise_path, pairwise_path = export_judge_tables(store, args.output_dir)
    report = evaluate_semantic_gate(
        pd.read_csv(pointwise_path),
        pd.read_csv(pairwise_path),
    )
    report.update(
        {
            "model_id": provider.model_id,
            "model_revision": provider.model_revision,
            "inference_config": getattr(provider, "inference_config", {}),
            "sample_csv": sample_path.name,
        }
    )
    report_path = args.output_dir / "semantic_pilot_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(
            "Semantic pilot failed. Do not start the full judge execution; "
            f"review {report_path}."
        )
    print(f"Semantic pilot passed: {report_path}")


if __name__ == "__main__":
    main()
