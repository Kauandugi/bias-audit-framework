from __future__ import annotations

from pathlib import Path

import pytest


streamlit = pytest.importorskip("streamlit")


def test_dashboard_switches_corpus_and_recomputes_statistics() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        Path(__file__).parents[1] / "app.py", default_timeout=60
    ).run()
    assert not app.error
    assert [(metric.label, metric.value) for metric in app.metric[:2]] == [
        ("Imagens", "64"),
        ("Rostos", "413"),
    ]
    assert set(app.dataframe[0].value["n_pairs"]) == {32}
    assert any(header.value == "Referência humana perceptual" for header in app.header)
    assert any(
        "Setenta participantes" in caption.value for caption in app.caption
    )

    app.selectbox[0].select("Replicação Vitor (64 imagens)").run()

    assert not app.error
    assert [(metric.label, metric.value) for metric in app.metric[:2]] == [
        ("Imagens", "64"),
        ("Rostos", "585"),
    ]
    wilcoxon = app.dataframe[0].value.set_index("dimensão")
    assert set(wilcoxon["n_pairs"]) == {32}
    assert wilcoxon.loc[
        "Diversidade de apresentação de gênero", "p_value_holm"
    ] == pytest.approx(0.007731183432042599)
