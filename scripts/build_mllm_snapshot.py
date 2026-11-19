"""Create a small deterministic source snapshot for the Colab MLLM workflow."""

from __future__ import annotations

import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "execucao" / "BiasAuditFW_mllm_validation_snapshot_v10.zip"

EXPLICIT_FILES = (
    "README.md",
    "pyproject.toml",
    "requirements-colab.txt",
    "notebooks/mllm_validation_colab.ipynb",
    "scripts/analyze_mllm_outputs.py",
    "scripts/build_mllm_validation_notebook.py",
    "scripts/run_mllm_semantic_pilot.py",
    "scripts/run_mllm_validation.py",
    "tests/conftest.py",
    "tests/test_human_mllm_statistics.py",
    "tests/test_judge_quality.py",
    "tests/test_judges.py",
    "tests/test_judges_external.py",
    "tests/test_mllm_notebook.py",
    "tests/test_sampling.py",
)


def snapshot_files() -> list[Path]:
    paths = [ROOT / relative for relative in EXPLICIT_FILES]
    paths.extend((ROOT / "src" / "biasauditfw").rglob("*.py"))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    return sorted(set(paths), key=lambda path: path.relative_to(ROOT).as_posix())


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in snapshot_files():
            relative = path.relative_to(ROOT).as_posix()
            info = ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(f"Wrote {OUTPUT}")
    print(f"SHA-256 {digest}")


if __name__ == "__main__":
    main()
