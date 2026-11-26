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
SCHEMA_DATASETS = {
    "Conjunto original (64 imagens)": DATA_DIR / "schema2",
    "Replicação parcial (64 imagens)": DATA_DIR / "schema2-vitor",
}
LEGACY_DATA = DATA_DIR / "resultados_auditoria_tcc.csv"
HUMAN_VALIDATION_DIR = DATA_DIR / "human-unesp-aggregate"
WIDER_SUMMARY = ROOT / "output" / "validation" / "wider-face" / "wider_detector_summary.csv"
MLLM_STATUS = ROOT / "output" / "validation" / "mllm" / "vitor-62" / "pilot_status.json"

METRICS = {
    "clip_racial_diversity_margin": "Diversidade racial percebida",
    "clip_gender_diversity_margin": "Diversidade de apresentação de gênero",
}

DIMENSION_LABELS = {
    "racial": "Diversidade racial percebida",
    "gender": "Equilíbrio de gênero percebido",
    "cultural": "Contexto latino-americano",
    "body": "Diversidade corporal",
    "quality": "Qualidade visual",
}

AUTOMATED_METRIC_LABELS = {
    "clip_racial_diversity_margin": "CLIP racial",
    "clip_gender_diversity_margin": "CLIP de gênero",
    "deepface_racial_simpson": "Gini-Simpson racial",
    "deepface_gender_simpson": "Gini-Simpson de gênero",
}

PROMPT_LABELS = {
    "Inclusive Prompts": "Inclusiva",
    "Neutral Prompts": "Neutra",
    "inclusive": "Inclusiva",
    "neutral": "Neutra",
}


def _legacy_image_id(value: str) -> str:
    return hashlib.sha256(value.casefold().encode("utf-8")).hexdigest()[:16]


@st.cache_data
def load_data(schema_data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Load schema 2.0, retaining the old table only as a visual legacy."""

    data_dir = Path(schema_data_dir)
    image_data = data_dir / "audit_images.csv"
    face_data = data_dir / "audit_faces.csv"
    if image_data.exists() and face_data.exists():
        images = pd.read_csv(image_data)
        faces = pd.read_csv(face_data)
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
def load_run_metadata(schema_data_dir: str) -> dict[str, object]:
    """Load provenance for the canonical schema 2.0 execution."""

    run_metadata = Path(schema_data_dir) / "run_metadata.json"
    if not run_metadata.exists():
        return {}
    with run_metadata.open(encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict):
        raise ValueError("run_metadata.json must contain a JSON object")
    return metadata


@st.cache_data
def load_human_validation(validation_dir: str) -> dict[str, pd.DataFrame]:
    """Load image-level human outputs only after an explicit ethics release."""

    root = Path(validation_dir)
    metadata_path = root / "human_validation_metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict):
        raise ValueError("human_validation_metadata.json must contain an object")
    if metadata.get("usable_as_final_evidence") is not True:
        return {}
    files = {
        "agreement": "human_interrater_agreement.csv",
        "prompt_effects": "human_paired_prompt_effects.csv",
        "convergence": "human_framework_convergence.csv",
        "participant_sensitivity": "human_participant_sensitivity.csv",
        "simpson": "deepface_simpson_paired.csv",
        "simpson_sensitivity": "deepface_simpson_sensitivity.csv",
    }
    loaded: dict[str, pd.DataFrame] = {}
    for key, filename in files.items():
        path = root / filename
        if path.exists():
            loaded[key] = pd.read_csv(path)
    return loaded


@st.cache_data
def load_wider_summary(path: str) -> dict[str, float]:
    """Load the overall WIDER FACE validation row."""

    summary_path = Path(path)
    if not summary_path.exists():
        return {}
    frame = pd.read_csv(summary_path)
    overall = frame.loc[frame["grupo"].eq("overall")]
    if overall.empty:
        return {}
    row = overall.iloc[0]
    return {
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1": float(row["f1"]),
        "mean_iou": float(row["mean_iou"]),
        "ap50": float(row["ap50_global"]),
    }


@st.cache_data
def load_json(path: str) -> dict[str, object]:
    """Load an optional public JSON artifact."""

    json_path = Path(path)
    if not json_path.exists():
        return {}
    with json_path.open(encoding="utf-8") as file:
        content = json.load(file)
    return content if isinstance(content, dict) else {}


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
    data["tipo_prompt"] = data["tipo_prompt"].map(
        lambda value: PROMPT_LABELS.get(str(value), str(value))
    )
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
    axis.legend(title="Tipo de instrução")
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)


st.set_page_config(
    page_title="Representação em imagens geradas por IA",
    layout="wide",
)

st.title("Representação em imagens geradas por IA")
st.caption(
    "Painel público do estudo sobre instruções neutras e inclusivas em cenas universitárias."
)
st.write(
    "Este painel ajuda a entender se pedir diversidade de forma explícita muda "
    "as imagens geradas. Ele reúne medidas automáticas, avaliação humana e "
    "testes de validação."
)
st.info(
    "Resposta principal: as instruções inclusivas mudaram alguns aspectos da "
    "cena, principalmente o contexto latino-americano e a diversidade corporal. "
    "Para diversidade racial e equilíbrio de gênero, a evidência principal foi "
    "fraca. Nenhuma medida isolada comprova diversidade ou ausência de viés."
)

with st.expander("Como ler este painel"):
    st.markdown(
        """
        * **Instrução neutra:** descreve apenas a cena, sem pedir diversidade.
        * **Instrução inclusiva:** pede diversidade e contexto latino-americano.
        * **CLIP:** compara a imagem com frases de referência. Ele mede proximidade semântica, não conta pessoas.
        * **DeepFace:** organiza os rostos detectados em categorias previstas pelo modelo.
        * **Avaliação humana:** mostra como os participantes perceberam as imagens.

        As categorias automáticas não definem identidade, ancestralidade, nacionalidade ou sexo biológico.
        """
    )

available_datasets = {
    label: path
    for label, path in SCHEMA_DATASETS.items()
    if (path / "audit_images.csv").exists() and (path / "audit_faces.csv").exists()
}

st.sidebar.header("Filtros")
selected_dataset = st.sidebar.selectbox(
    "Conjunto de imagens",
    list(available_datasets) or list(SCHEMA_DATASETS),
)
selected_data_dir = available_datasets.get(
    selected_dataset,
    SCHEMA_DATASETS[selected_dataset],
)

try:
    images_df, faces_df, using_legacy = load_data(str(selected_data_dir))
    run_metadata = (
        {} if using_legacy else load_run_metadata(str(selected_data_dir))
    )
except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError) as exc:
    st.error(f"Não foi possível validar os arquivos do painel: {exc}")
    st.stop()

if images_df.empty:
    st.error(
        f"Nenhum resultado foi encontrado em {selected_data_dir.relative_to(ROOT)}. "
        "Execute o pipeline e gere audit_images.csv e audit_faces.csv."
    )
    st.stop()

if using_legacy:
    st.warning(
        "Este conjunto usa o formato antigo e está disponível apenas para consulta histórica."
    )

models = sorted(images_df["modelo_ia"].dropna().astype(str).unique())
selected_models = st.sidebar.multiselect(
    "Geradores", models, default=models
)
prompts = sorted(images_df["tipo_prompt"].dropna().astype(str).unique())
selected_prompts = st.sidebar.multiselect(
    "Tipos de instrução",
    prompts,
    default=prompts,
    format_func=lambda value: PROMPT_LABELS.get(str(value), str(value)),
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
    image_metadata = filtered_images[
        ["imagem_id", "tipo_prompt", "modelo_ia"]
    ].drop_duplicates("imagem_id")
    filtered_faces = filtered_faces.drop(
        columns=["tipo_prompt", "modelo_ia"], errors="ignore"
    ).merge(image_metadata, on="imagem_id", how="left", validate="many_to_one")

human_outputs = load_human_validation(str(HUMAN_VALIDATION_DIR))
wider_summary = load_wider_summary(str(WIDER_SUMMARY))
mllm_status = load_json(str(MLLM_STATUS))

pair_count = (
    filtered_images["pair_id"].nunique()
    if "pair_id" in filtered_images
    else len(filtered_images) // 2
)
coverage = 100.0 * (filtered_images["num_faces"].fillna(0) > 0).mean()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Imagens", filtered_images["imagem_id"].nunique())
col2.metric("Rostos detectados", len(filtered_faces))
col3.metric("Pares comparáveis", pair_count)
col4.metric("Imagens com rosto", f"{coverage:.1f}%")

prompt_face_counts = (
    filtered_images.groupby("tipo_prompt", observed=True)["num_faces"]
    .sum()
    .to_dict()
)
neutral_faces = next(
    (value for key, value in prompt_face_counts.items() if "neutr" in str(key).lower()),
    None,
)
inclusive_faces = next(
    (value for key, value in prompt_face_counts.items() if "inclus" in str(key).lower()),
    None,
)
if neutral_faces is not None and inclusive_faces is not None:
    st.subheader("Primeira leitura do conjunto")
    st.write(
        f"O detector encontrou **{int(neutral_faces)} rostos** nas imagens neutras "
        f"e **{int(inclusive_faces)} rostos** nas imagens inclusivas. A diferença "
        "mostra que as imagens inclusivas costumam ter mais pessoas, mas não diz "
        "sozinha se a representação ficou mais diversa."
    )

sns.set_theme(style="whitegrid", palette="muted")
automatic_tab, human_tab, validation_tab, data_tab = st.tabs(
    [
        "Medidas automáticas",
        "Avaliação humana",
        "Validação e limites",
        "Dados e reprodução",
    ]
)

with automatic_tab:
    st.header("O que as medidas automáticas mostram")
    st.write(
        "O CLIP observa a imagem inteira. RetinaFace e DeepFace trabalham com "
        "os rostos detectados. Por medirem coisas diferentes, os resultados não "
        "precisam apontar na mesma direção."
    )

    st.subheader("Categorias previstas nos rostos")
    st.caption(
        "Os gráficos mostram a distribuição das categorias previstas pelo DeepFace. "
        "Eles não mostram a identidade real das pessoas."
    )
    if filtered_faces.empty:
        st.info("Não há rostos detectados para os filtros selecionados.")
    else:
        left, right = st.columns(2)
        with left:
            render_face_chart(
                filtered_faces,
                "raca",
                "Categorias raciais previstas",
                "Categoria prevista pelo DeepFace",
            )
        with right:
            render_face_chart(
                filtered_faces,
                "genero",
                "Categorias de gênero previstas",
                "Categoria prevista pelo DeepFace",
            )

    st.subheader("Comparação semântica com CLIP")
    st.caption(
        "Uma margem positiva indica maior proximidade com as frases de diversidade. "
        "O valor não é uma porcentagem e não representa uma contagem de pessoas."
    )
    available_metrics = [metric for metric in METRICS if metric in filtered_images]
    if len(available_metrics) != len(METRICS):
        st.info("Este conjunto não contém as duas medidas do CLIP.")
    else:
        metric_tabs = st.tabs(list(METRICS.values()))
        for metric_tab, metric in zip(metric_tabs, available_metrics):
            with metric_tab:
                chart_data = filtered_images.copy()
                chart_data["tipo_prompt"] = chart_data["tipo_prompt"].map(
                    lambda value: PROMPT_LABELS.get(str(value), str(value))
                )
                figure, axis = plt.subplots(figsize=(9, 4))
                sns.boxplot(
                    data=chart_data,
                    x="tipo_prompt",
                    y=metric,
                    hue="modelo_ia",
                    showmeans=True,
                    ax=axis,
                )
                axis.axhline(0, color="black", linewidth=0.8)
                axis.set_xlabel("Tipo de instrução")
                axis.set_ylabel("Margem do CLIP")
                axis.legend(title="Gerador")
                figure.tight_layout()
                st.pyplot(figure)
                plt.close(figure)

        eligible, reason = inference_eligibility(filtered_images)
        if eligible:
            paired_report = wilcoxon_reports(filtered_images)
            paired_report["dimensão"] = paired_report["metric"].map(METRICS)
            with st.expander("Detalhes estatísticos do CLIP"):
                st.write(
                    "A diferença foi calculada como imagem inclusiva menos imagem "
                    "neutra. Um valor positivo favorece a imagem inclusiva."
                )
                st.dataframe(
                    paired_report[
                        [
                            "dimensão",
                            "n_pairs",
                            "median_paired_difference",
                            "n_inclusive_higher",
                            "n_neutral_higher",
                            "n_ties",
                            "p_value_holm",
                        ]
                    ],
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "dimensão": "Dimensão",
                        "n_pairs": "Pares",
                        "median_paired_difference": "Diferença mediana",
                        "n_inclusive_higher": "Inclusiva maior",
                        "n_neutral_higher": "Neutra maior",
                        "n_ties": "Empates",
                        "p_value_holm": "p ajustado por Holm",
                    },
                )

                sensitivity_report = mann_whitney_reports(filtered_images)
                sensitivity_report["dimensão"] = sensitivity_report["metric"].map(
                    METRICS
                )
                st.caption(
                    "A tabela abaixo é uma análise secundária que não usa o pareamento."
                )
                st.dataframe(
                    sensitivity_report[
                        [
                            "dimensão",
                            "n_inclusive",
                            "n_neutral",
                            "p_value_holm",
                            "rank_biserial",
                        ]
                    ],
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "dimensão": "Dimensão",
                        "n_inclusive": "Imagens inclusivas",
                        "n_neutral": "Imagens neutras",
                        "p_value_holm": "p ajustado por Holm",
                        "rank_biserial": "Efeito bisserial",
                    },
                )
                st.info(
                    "Um p maior ou igual a 0,05 indica evidência insuficiente de "
                    "diferença. Isso não prova que os grupos sejam iguais."
                )
        else:
            st.warning(f"A análise estatística não foi executada: {reason}.")

    if "simpson" in human_outputs:
        framework_corpus = (
            "original-64"
            if selected_dataset == "Conjunto original (64 imagens)"
            else "vitor-64"
        )
        simpson = human_outputs["simpson"]
        simpson = simpson.loc[simpson["dataset_id"].eq(framework_corpus)].copy()
        simpson["dimension"] = simpson["dimension"].map(DIMENSION_LABELS)
        st.subheader("Variedade das categorias faciais")
        st.write(
            "O índice de Gini-Simpson aumenta quando as categorias previstas ficam "
            "menos concentradas. Ele só é calculado em imagens com pelo menos dois rostos."
        )
        st.dataframe(
            simpson[
                [
                    "dimension",
                    "n_pairs",
                    "median_paired_difference",
                    "median_difference_ci_2_5",
                    "median_difference_ci_97_5",
                    "p_value_holm",
                ]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "dimension": "Dimensão",
                "n_pairs": "Pares válidos",
                "median_paired_difference": "Diferença mediana",
                "median_difference_ci_2_5": "Limite inferior",
                "median_difference_ci_97_5": "Limite superior",
                "p_value_holm": "p ajustado por Holm",
            },
        )

with human_tab:
    st.header("O que os participantes perceberam")
    st.write(
        "Setenta participantes avaliaram 48 imagens em cinco dimensões. Isso "
        "gerou 3.360 avaliações de imagens e 16.800 notas. O painel usa somente "
        "resultados agregados."
    )
    human_corpus = (
        "original-64"
        if selected_dataset == "Conjunto original (64 imagens)"
        else "vitor-form-24"
    )
    framework_corpus = (
        "original-64"
        if selected_dataset == "Conjunto original (64 imagens)"
        else "vitor-64"
    )

    if "prompt_effects" in human_outputs:
        effects = human_outputs["prompt_effects"]
        effects = effects.loc[effects["human_corpus_id"].eq(human_corpus)].copy()
        effects["dimension"] = effects["dimension"].map(DIMENSION_LABELS)
        effects["analysis_role"] = effects["analysis_role"].map(
            {"primary": "Principal", "exploratory": "Exploratória"}
        )
        st.subheader("Comparação entre imagens inclusivas e neutras")
        st.caption(
            "Valor positivo indica nota maior para a imagem inclusiva. Cada linha usa 12 pares."
        )
        st.dataframe(
            effects[
                [
                    "dimension",
                    "analysis_role",
                    "n_pairs",
                    "median_paired_difference",
                    "median_difference_ci_2_5",
                    "median_difference_ci_97_5",
                    "p_value_holm",
                ]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "dimension": "Dimensão avaliada",
                "analysis_role": "Papel na análise",
                "n_pairs": "Pares",
                "median_paired_difference": "Diferença mediana",
                "median_difference_ci_2_5": "Limite inferior",
                "median_difference_ci_97_5": "Limite superior",
                "p_value_holm": "p ajustado por Holm",
            },
        )

    if "agreement" in human_outputs:
        agreement = human_outputs["agreement"].copy()
        agreement["dimension"] = agreement["dimension"].map(DIMENSION_LABELS)
        st.subheader("Concordância entre participantes")
        st.write(
            "Os valores ficaram baixos em todas as dimensões. Isso significa que "
            "os participantes não formaram uma opinião uniforme sobre as imagens."
        )
        st.dataframe(
            agreement[["dimension", "n_raters", "krippendorff_alpha_ordinal"]],
            hide_index=True,
            width="stretch",
            column_config={
                "dimension": "Dimensão avaliada",
                "n_raters": "Participantes",
                "krippendorff_alpha_ordinal": "Alfa de Krippendorff",
            },
        )

    if "convergence" in human_outputs:
        convergence = human_outputs["convergence"]
        convergence = convergence.loc[
            convergence["dataset_id"].eq(framework_corpus)
        ].copy()
        convergence["dimension"] = convergence["dimension"].map(DIMENSION_LABELS)
        convergence["automated_metric"] = convergence["automated_metric"].map(
            AUTOMATED_METRIC_LABELS
        )
        st.subheader("Comparação entre pessoas e ferramenta")
        st.write(
            "As notas humanas foram cruzadas com as medidas automáticas da mesma "
            "imagem. As medidas do CLIP não acompanharam as notas humanas."
        )
        st.dataframe(
            convergence[
                [
                    "dimension",
                    "automated_metric",
                    "n",
                    "spearman_rho",
                    "spearman_ci_2_5",
                    "spearman_ci_97_5",
                    "spearman_p_value_holm",
                    "partial_spearman_rho_face_count",
                    "partial_spearman_p_value_face_count_holm",
                ]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "dimension": "Dimensão humana",
                "automated_metric": "Medida automática",
                "n": "Imagens",
                "spearman_rho": "Correlação simples",
                "spearman_ci_2_5": "Limite inferior",
                "spearman_ci_97_5": "Limite superior",
                "spearman_p_value_holm": "p ajustado",
                "partial_spearman_rho_face_count": "Correlação com controle de rostos",
                "partial_spearman_p_value_face_count_holm": "p ajustado com controle",
            },
        )
        st.info(
            "A avaliação humana é uma referência perceptual. Nem as respostas, "
            "nem o CLIP, nem o DeepFace constituem verdade demográfica."
        )

with validation_tab:
    st.header("Validação técnica e limites")
    st.write(
        "Os testes desta seção ajudam a entender onde a ferramenta funciona bem "
        "e onde seus resultados exigem cautela."
    )

    if wider_summary:
        st.subheader("RetinaFace no WIDER FACE")
        st.write(
            "O detector quase não marcou rostos inexistentes, mas deixou de "
            "encontrar parte dos rostos anotados. Esse limite pode ser mais "
            "importante em cenas com muitas pessoas."
        )
        wider1, wider2, wider3, wider4 = st.columns(4)
        wider1.metric("Precisão", f"{wider_summary['precision']:.3f}")
        wider2.metric("Revocação", f"{wider_summary['recall']:.3f}")
        wider3.metric("F1", f"{wider_summary['f1']:.3f}")
        wider4.metric("AP50 global", f"{wider_summary['ap50']:.3f}")
        st.caption(
            "A precisão alta indica poucos falsos positivos. A revocação de 0,567 "
            "indica que cerca de 43% dos rostos anotados não foram encontrados."
        )

    st.subheader("Pilotos com modelos multimodais")
    if mllm_status:
        st.warning(
            "Os pilotos não atingiram a cobertura e a estabilidade necessárias. "
            "Por isso, não foram usados nas conclusões do estudo."
        )
    else:
        st.info("O relatório público dos pilotos multimodais não está disponível.")

    st.subheader("Limites que devem acompanhar a leitura")
    st.markdown(
        """
        * Cada combinação de cena e gerador tem apenas uma imagem.
        * A avaliação humana cobre 12 pares de cada conjunto.
        * A concordância entre participantes foi baixa.
        * Os classificadores podem discordar sobre o mesmo rosto.
        * O CLIP depende das frases usadas como referência.
        * Os resultados descrevem estas imagens e não todos os geradores possíveis.
        """
    )

with data_tab:
    st.header("Dados públicos e reprodução")
    st.write(
        "O repositório publica o código, os manifestos, os resultados automáticos "
        "e os dados humanos agregados. Respostas individuais, perfis, consentimentos "
        "e imagens-fonte permanecem fora do repositório."
    )

    download_left, download_right = st.columns(2)
    with download_left:
        st.download_button(
            "Baixar resultados por imagem",
            filtered_images.to_csv(index=False).encode("utf-8"),
            file_name="resultados_por_imagem.csv",
            mime="text/csv",
        )
    with download_right:
        st.download_button(
            "Baixar resultados por rosto",
            filtered_faces.to_csv(index=False).encode("utf-8"),
            file_name="resultados_por_rosto.csv",
            mime="text/csv",
        )

    if run_metadata:
        with st.expander("Informações da execução"):
            raw_timestamp = pd.to_datetime(
                run_metadata.get("run_timestamp_utc"), errors="coerce", utc=True
            )
            timestamp = (
                raw_timestamp.strftime("%d/%m/%Y %H:%M UTC")
                if not pd.isna(raw_timestamp)
                else "não registrada"
            )
            st.write(f"Formato dos dados: {run_metadata.get('schema_version', '2.0')}")
            st.write(f"Data da execução: {timestamp}")
            st.write(f"Modo: {run_metadata.get('run_mode', 'não registrado')}")
            st.write(f"GPU: {run_metadata.get('gpu_name', 'não registrada')}")
            st.write(
                "Revisão do código: "
                f"{str(run_metadata.get('repository_commit', 'não registrada'))[:12]}"
            )
