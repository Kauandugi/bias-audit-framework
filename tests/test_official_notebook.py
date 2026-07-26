from __future__ import annotations

from pathlib import Path

import nbformat


NOTEBOOK = Path(__file__).parents[1] / "Framework_Auditoria_Viés_IA_Generativa.ipynb"


def test_official_notebook_is_clean_schema_2_orchestrator() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    text = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
    )

    assert len(notebook["cells"]) == 28
    assert "ClipDiversityScorer" in text
    assert "discover_dataset(CONFIG)" in text
    assert "validate_full_outputs.py" in text
    assert 'REPOSITORY_REF = "agent/schema-2-diversity-refactor"' in text
    assert '"checkout", "--quiet", "--force", "FETCH_HEAD"' in text
    assert '"rev-parse", "HEAD"' in text
    assert "import biasauditfw" in text
    assert "BiasAuditFW schema {SCHEMA_VERSION}" in text
    assert "clip_latin_american" not in text
    assert "clip_not_latin_american" not in text
    assert "results[0]" not in text
    assert "parts[0]" not in text
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
