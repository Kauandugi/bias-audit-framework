from __future__ import annotations

from pathlib import Path
import tomllib

import pytest


streamlit = pytest.importorskip("streamlit")


def test_streamlit_cloud_dependencies_are_core_dependencies() -> None:
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as file:
        pyproject = tomllib.load(file)

    dependencies = pyproject["project"]["dependencies"]
    for package in ("matplotlib", "seaborn", "streamlit"):
        assert any(
            dependency.lower().startswith(package) for dependency in dependencies
        ), f"{package} must be installed by the default Streamlit Cloud environment"


def test_dashboard_switches_corpus_and_recomputes_statistics() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        Path(__file__).parents[1] / "app.py", default_timeout=60
    ).run()
    assert not app.error
    assert [(metric.label, metric.value) for metric in app.metric[:2]] == [
        ("Imagens", "64"),
        ("Rostos detectados", "413"),
    ]
    assert set(app.dataframe[0].value["n_pairs"]) == {32}
    assert any(
        header.value == "O que os participantes perceberam" for header in app.header
    )
    assert any(
        "Setenta participantes" in block.value for block in app.markdown
    )
    assert "DALL-E" in app.multiselect[0].options

    app.selectbox[0].select("Replicação parcial (64 imagens)").run()

    assert not app.error
    assert "DALL-E" in app.multiselect[0].options
    assert "OpenAI image_gen via Codex" not in app.multiselect[0].options
    assert [(metric.label, metric.value) for metric in app.metric[:2]] == [
        ("Imagens", "64"),
        ("Rostos detectados", "585"),
    ]
    wilcoxon = app.dataframe[0].value.set_index("dimensão")
    assert set(wilcoxon["n_pairs"]) == {32}
    assert wilcoxon.loc[
        "Diversidade de apresentação de gênero", "p_value_holm"
    ] == pytest.approx(0.007731183432042599)
