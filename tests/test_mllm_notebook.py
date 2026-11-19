from __future__ import annotations

import hashlib
from pathlib import Path

import nbformat


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "mllm_validation_colab.ipynb"


def test_mllm_notebook_has_two_gated_local_judges() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    text = "\n".join(cell.source for cell in notebook.cells)

    assert "Qwen/Qwen3-VL-8B-Instruct" in text
    assert "google/gemma-3-4b-it" in text
    assert "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b" in text
    assert "093f9f388b31de276ce2de164bdc2081324b9767" in text
    assert 'userdata.get("HF_TOKEN")' in text
    assert 'os.environ["HF_TOKEN"] = hf_token' in text
    assert '"--provider",\n        provider_name' in text
    assert "RUN_FULL_QWEN3 = False" in text
    assert "RUN_FULL_GEMMA = False" in text
    assert "FULL_SAMPLE_CSV = None" in text
    assert '"--sample-csv"' in text
    assert "Set FULL_SAMPLE_CSV" in text
    assert '"--qwen3-revision"' in text
    assert '"--gemma-revision"' in text
    assert text.count('"4bit-nf4"') >= 2
    assert "if not report[\"passed\"]" in text
    assert "Human participant records must not be" in text

    for cell in notebook.cells:
        payload = f"{cell.cell_type}\0{cell.source}".encode("utf-8")
        assert cell.id == hashlib.sha256(payload).hexdigest()[:12]
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert cell.outputs == []
