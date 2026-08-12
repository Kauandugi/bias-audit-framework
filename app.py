"""Streamlit dashboard for BiasAuditFW schema 2.0 artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from biasauditfw.contracts import validate_contract  # noqa: E402
from biasauditfw.statistics import (  # noqa: E402
    inference_eligibility,
    mann_whitney_reports,
    wilcoxon_reports,
)


DATA_DIR = ROOT / "data"
SCHEMA_DATA_DIR = DATA_DIR / "schema2"
IMAGE_DATA = SCHEMA_DATA_DIR / "audit_images.csv"
FACE_DATA = SCHEMA_DATA_DIR / "audit_faces.csv"
RUN_METADATA = SCHEMA_DATA_DIR / "run_metadata.json"
LEGACY_DATA = DATA_DIR / "resultados_auditoria_tcc.csv"

METRICS = {
    "clip_racial_diversity_margin": "Diversidade racial percebida",
    "clip_gender_diversity_margin": "Diversidade de apresentação de gênero",
}


def _legacy_image_id(value: str) -> str:
    return hashlib.sha256(value.casefold().encode("utf-8")).hexdigest()[:16]


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Load schema 2.0, retaining the old table only as a visual legacy."""

    if IMAGE_DATA.exists() and FACE_DATA.exists():
        images = pd.read_csv(IMAGE_DATA)
        faces = pd.read_csv(FACE_DATA)
        validate_contract(images, faces)
        return images, faces, False
    if not LEGACY_DATA.exists():
        return pd.DataFrame(), pd.DataFrame(), False

    legacy = pd.read_csv(LEGACY_DATA).copy()
    legacy["imagem_id"] = legacy["nome_arquivo"].astype(str).map(_legacy_image_id)
    if "modelo_ia" not in legacy:
        legacy["modelo_ia"] = "Modelo não registrado"
    legacy["num_faces"] = legacy["raca"].notna().astype(int)
    image_columns = [
        "imagem_id",
        "nome_arquivo",
        "tipo_prompt",
        "modelo_ia",
        "num_faces",
    ]
    images = legacy[image_columns].copy()
    faces = legacy[
        ["imagem_id", "raca", "genero", "tipo_prompt", "modelo_ia"]
    ].dropna(subset=["raca", "genero"], how="all")
    return images, faces, True


@st.cache_data
def load_run_metadata() -> dict[str, object]:
    """Load provenance for the canonical schema 2.0 execution."""

    if not RUN_METADATA.exists():
        return {}
    with RUN_METADATA.open(encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict):
        raise ValueError("run_metadata.json must contain a JSON object")
    return metadata


def face_category_proportions(
    faces: pd.DataFrame, category: str
) -> pd.DataFrame:
    """Calculate within-prompt percentages for face-level categories."""

    valid = faces.dropna(subset=["tipo_prompt", category]).copy()
    counts = (
        valid.groupby(["tipo_prompt", category], observed=True)
        .size()
        .rename("contagem")
        .reset_index()
    )
    totals = counts.groupby("tipo_prompt", observed=True)["contagem"].transform("sum")
    counts["percentual"] = 100.0 * counts["contagem"] / totals
    return counts


def render_face_chart(
    faces: pd.DataFrame,
    category: str,
    title: str,
    axis_label: str,
) -> None:
    st.subheader(title)
    data = face_category_proportions(faces, category)
    if data.empty:
        st.info("Não há valores válidos para este gráfico.")
        return
    figure, axis = plt.subplots(figsize=(7, 4))
    order = faces[category].dropna().value_counts().index
    sns.barplot(
        data=data,
        x=category,
        y="percentual",
        hue="tipo_prompt",
        order=order,
        ax=axis,
    )
    axis.set_xlabel(axis_label)
    axis.set_ylabel("Percentual dentro do grupo de prompt (%)")
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)


st.set_page_config(page_title="BiasAuditFW", layout="wide")
st.title("BiasAuditFW: análise exploratória de representação")
st.caption(
    "CLIP opera no nível da imagem; DeepFace opera no nível de cada rosto detectado."
)

try:
    images_df, faces_df, using_legacy = load_data()
    run_metadata = {} if using_legacy else load_run_metadata()
except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError) as exc:
    st.error(f"Falha ao validar os artefatos do dashboard: {exc}")
    st.stop()

if images_df.empty:
    st.error(
        "Nenhum artefato foi encontrado em data/schema2/. Execute o notebook "
        "schema 2.0 e copie audit_images.csv e audit_faces.csv."
    )
    st.stop()

if using_legacy:
    st.warning(
        "O dashboard está exibindo o CSV legado. Ele serve apenas para inspeção "
        "histórica e não contém os novos escores de diversidade."
    )
elif run_metadata:
    raw_timestamp = pd.to_datetime(
        run_metadata.get("run_timestamp_utc"), errors="coerce", utc=True
    )
    timestamp = (
        raw_timestamp.strftime("%Y-%m-%d %H:%M UTC")
        if not pd.isna(raw_timestamp)
        else "data não registrada"
    )
    commit = str(run_metadata.get("repository_commit", "não registrado"))[:7]
    st.caption(
        " | ".join(
            [
                f"Schema {run_metadata.get('schema_version', '2.0')}",
                f"execução {run_metadata.get('run_mode', 'não registrada')}",
                timestamp,
                str(run_metadata.get("gpu_name", "GPU não registrada")),
                f"revisão {commit}",
            ]
        )
    )

models = sorted(images_df["modelo_ia"].dropna().astype(str).unique())
selected_models = st.sidebar.multiselect(
    "Modelos geradores", models, default=models
)
prompts = sorted(images_df["tipo_prompt"].dropna().astype(str).unique())
selected_prompts = st.sidebar.multiselect(
    "Tipos de prompt", prompts, default=prompts
)

model_mask = images_df["modelo_ia"].astype(str).isin(selected_models)
prompt_mask = images_df["tipo_prompt"].astype(str).isin(selected_prompts)
filtered_images = images_df.loc[model_mask & prompt_mask].copy()
selected_image_ids = set(filtered_images["imagem_id"])
filtered_faces = faces_df.loc[
    faces_df["imagem_id"].isin(selected_image_ids)
].copy()

if filtered_images.empty:
    st.info("Nenhuma imagem corresponde aos filtros selecionados.")
    st.stop()

if not filtered_faces.empty:
    metadata = filtered_images[
        ["imagem_id", "tipo_prompt", "modelo_ia"]
    ].drop_duplicates("imagem_id")
    filtered_faces = filtered_faces.drop(
        columns=["tipo_prompt", "modelo_ia"], errors="ignore"
    ).merge(metadata, on="imagem_id", how="left", validate="many_to_one")

coverage = 100.0 * (filtered_images["num_faces"].fillna(0) > 0).mean()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Imagens", filtered_images["imagem_id"].nunique())
col2.metric("Rostos", len(filtered_faces))
col3.metric("Rostos por imagem", f"{len(filtered_faces) / len(filtered_images):.2f}")
col4.metric("Cobertura de detecção", f"{coverage:.1f}%")

sns.set_theme(style="whitegrid", palette="muted")
st.divider()
st.header("Composição facial predita")
st.caption(
    "As categorias descrevem saídas dos classificadores sobre faces sintéticas; "
    "não representam identidade, ancestralidade ou sexo biológico."
)
if filtered_faces.empty:
    st.info("Não há rostos detectados para os filtros selecionados.")
else:
    left, right = st.columns(2)
    with left:
        render_face_chart(
            filtered_faces,
            "raca",
            "Categorias raciais percebidas",
            "Predição do DeepFace",
        )
    with right:
        render_face_chart(
            filtered_faces,
            "genero",
            "Apresentação de gênero percebida",
            "Predição do DeepFace",
        )

st.divider()
st.header("Diversidade semântica no nível da imagem")
st.caption(
    "Cada margem é cos(diversidade) − cos(homogeneidade), após normalização L2. "
    "Ela mede alinhamento semântico, não contagem ou proporção de pessoas."
)

available_metrics = [metric for metric in METRICS if metric in filtered_images]
if len(available_metrics) != len(METRICS):
    st.info(
        "O conjunto carregado não contém os dois escores CLIP do schema 2.0."
    )
else:
    tabs = st.tabs(list(METRICS.values()))
    for tab, metric in zip(tabs, available_metrics):
        with tab:
            figure, axis = plt.subplots(figsize=(9, 4))
            sns.boxplot(
                data=filtered_images,
                x="tipo_prompt",
                y=metric,
                hue="modelo_ia",
                showmeans=True,
                ax=axis,
            )
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_xlabel("Tipo de prompt")
            axis.set_ylabel("Margem CLIP")
            figure.tight_layout()
            st.pyplot(figure)
            plt.close(figure)

    eligible, reason = inference_eligibility(filtered_images)
    if eligible:
        report = mann_whitney_reports(filtered_images)
        report["dimensão"] = report["metric"].map(METRICS)
        st.subheader("Mann–Whitney com correção de Holm")
        st.dataframe(
            report[
                [
                    "dimensão",
                    "n_inclusive",
                    "n_neutral",
                    "u_statistic",
                    "p_value",
                    "p_value_holm",
                    "rank_biserial",
                    "decision_holm",
                ]
            ],
            hide_index=True,
            width="stretch",
        )

        paired_report = wilcoxon_reports(filtered_images)
        paired_report["dimensão"] = paired_report["metric"].map(METRICS)
        st.subheader("Wilcoxon pareado: análise de sensibilidade")
        st.dataframe(
            paired_report[
                [
                    "dimensão",
                    "n_pairs",
                    "median_paired_difference",
                    "n_inclusive_higher",
                    "n_neutral_higher",
                    "n_ties",
                    "w_statistic",
                    "p_value",
                    "p_value_holm",
                ]
            ],
            hide_index=True,
            width="stretch",
        )
        st.info(
            "“Não se rejeita H0” indica evidência insuficiente de diferença; "
            "não demonstra igualdade entre os grupos."
        )
    else:
        st.warning(f"Inferência estatística não executada: {reason}.")
