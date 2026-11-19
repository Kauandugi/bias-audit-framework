"""Run optional MLLM judges after the canonical BiasAuditFW pipeline."""

from __future__ import annotations

import argparse
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
    export_judge_tables,
    run_pairwise,
    run_pointwise,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-csv", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=("gemini", "groq", "qwen", "qwen3", "gemma"),
        default=("qwen3", "gemma"),
    )
    parser.add_argument(
        "--mode", choices=("pointwise", "pairwise", "both"), default="both"
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--sample-csv", type=Path)
    parser.add_argument("--limit", type=int)
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


def providers(names: list[str], args: argparse.Namespace) -> list[object]:
    configured = {
        "gemini": lambda: GeminiProvider(model_id=args.gemini_model_id),
        "groq": lambda: GroqProvider(model_id=args.groq_model_id),
        "qwen": lambda: LocalQwenProvider(
            model_id=args.qwen_model_id,
            model_revision=args.qwen_revision,
            min_pixels=args.qwen_min_visual_tokens * 28 * 28,
            max_pixels=args.qwen_max_visual_tokens * 28 * 28,
            max_new_tokens=args.qwen_max_new_tokens,
            quantization=args.qwen_quantization,
        ),
        "qwen3": lambda: LocalQwen3VLProvider(
            model_id=args.qwen3_model_id,
            model_revision=args.qwen3_revision,
            max_new_tokens=args.qwen3_max_new_tokens,
            quantization=args.qwen3_quantization,
        ),
        "gemma": lambda: LocalGemmaProvider(
            model_id=args.gemma_model_id,
            model_revision=args.gemma_revision,
            max_new_tokens=args.gemma_max_new_tokens,
            quantization=args.gemma_quantization,
        ),
    }
    return [configured[name]() for name in names]


def main() -> None:
    args = parse_args()
    images = pd.read_csv(args.images_csv)
    selected = images
    if args.sample_csv:
        sample = pd.read_csv(args.sample_csv)
        if "imagem_id" not in sample:
            raise ValueError("sample CSV must contain imagem_id")
        selected = images[images["imagem_id"].isin(sample["imagem_id"])].copy()
        if len(selected) != sample["imagem_id"].nunique():
            raise ValueError("sample CSV contains imagem_id values absent from audit_images")
    if args.limit:
        selected = selected.head(args.limit).copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = JudgeRunStore(args.output_dir / "judge_runs.jsonl")
    configured = providers(args.providers, args)
    if args.mode in {"pointwise", "both"}:
        run_pointwise(
            configured,
            selected,
            args.input_root,
            store,
            repeats=args.repeats,
        )
    if args.mode in {"pairwise", "both"}:
        if args.limit:
            raise ValueError("pairwise mode cannot use --limit because it may split pairs")
        run_pairwise(
            configured,
            build_pair_tasks(selected, args.input_root),
            store,
        )
    paths = export_judge_tables(store, args.output_dir)
    print(f"Pointwise table: {paths[0]}")
    print(f"Pairwise table: {paths[1]}")


if __name__ == "__main__":
    main()
