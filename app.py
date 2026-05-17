import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(page_title="BiasAuditFW Dashboard", layout="wide", page_icon="🔍")

st.title("🔍 BiasAuditFW: Auditoria de Viés em IA Generativa")
st.markdown("Dashboard interativo para análise de vieses demográficos e culturais em modelos de difusão texto-imagem.")

# ==========================================
# 2. CARREGAMENTO E PROTEÇÃO DE DADOS
# ==========================================
@st.cache_data
def load_data():
    try:
        # Lê o banco de dados oficial validado
        df = pd.read_csv('resultados_auditoria_tcc.csv')
        return df
    except FileNotFoundError:
        st.error("⚠️ Arquivo 'resultados_auditoria_tcc.csv' não encontrado. Verifique se ele está no repositório.")
        return pd.DataFrame()

df = load_data()

# Só executa o resto do app se os dados foram carregados com sucesso
if not df.empty:
    
    # Tratamento de segurança caso a coluna 'modelo_ia' não exista no CSV atual
    if 'modelo_ia' not in df.columns:
        df['modelo_ia'] = 'Modelo Desconhecido'

    # ==========================================
    # 3. BARRA LATERAL (FILTROS)
    # ==========================================
    st.sidebar.header("⚙️ Filtros de Análise")
    modelos_disponiveis = df['modelo_ia'].dropna().unique().tolist()
    
    # Cria um menu de múltipla escolha para o avaliador brincar com os dados
    modelo_selecionado = st.sidebar.multiselect(
        "Filtrar por Modelo de IA:", 
        options=modelos_disponiveis, 
        default=modelos_disponiveis
    )

    # Aplica o filtro aos dados
    df_filtrado = df[df['modelo_ia'].isin(modelo_selecionado)]

    # ==========================================
    # 4. MÉTRICAS PRINCIPAIS (KPIs)
    # ==========================================
    st.subheader("📊 Visão Geral")
    col1, col2, col3 = st.columns(3)
    
    col1.metric(label="Imagens Analisadas", value=len(df_filtrado))
    
    # Conta instâncias onde a raça predominante foi 'white'
    brancos = len(df_filtrado[df_filtrado['raca'] == 'white'])
    col2.metric(label="Rostos Brancos Detectados", value=brancos)
    
    # Conta instâncias onde o gênero predominante foi 'Man'
    homens = len(df_filtrado[df_filtrado['genero'] == 'Man'])
    col3.metric(label="Rostos Masculinos Detectados", value=homens)

    st.divider()

    # ==========================================
    # 5. GRÁFICOS DEMOGRÁFICOS
    # ==========================================
    sns.set_theme(style="whitegrid", palette="muted")
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Distribuição Racial")
        fig_raca, ax1 = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df_filtrado, x='raca', hue='tipo_prompt', 
                      order=df_filtrado['raca'].value_counts().index, ax=ax1)
        plt.xticks(rotation=45)
        plt.xlabel("Raça Predita")
        plt.ylabel("Frequência Absoluta")
        st.pyplot(fig_raca)

    with colB:
        st.subheader("Distribuição de Gênero")
        fig_genero, ax2 = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df_filtrado, x='genero', hue='tipo_prompt', ax=ax2)
        plt.xlabel("Gênero Predito")
        plt.ylabel("Frequência Absoluta")
        st.pyplot(fig_genero)

    # ==========================================
    # 6. ANÁLISE SEMÂNTICA E ESTATÍSTICA (CLIP)
    # ==========================================
    st.divider()
    st.subheader("🌍 Análise Semântica (CLIP) e Teste de Hipótese")
    ancora_latina = 'A typical Latin American university campus'
    
    # Verifica se a coluna de similaridade cultural existe
    if ancora_latina in df_filtrado.columns:
        fig_clip, ax3 = plt.subplots(figsize=(10, 4))
        sns.boxplot(data=df_filtrado, x='tipo_prompt', y=ancora_latina, showmeans=True, ax=ax3)
        plt.ylabel("Similaridade de Cosseno")
        plt.xlabel("Cenário de Teste")
        st.pyplot(fig_clip)

        # Prepara os dados para o Teste Estatístico
        inclusivo = df_filtrado[df_filtrado['tipo_prompt'] == 'Inclusive Prompts'][ancora_latina].dropna()
        neutro = df_filtrado[df_filtrado['tipo_prompt'] == 'Neutral Prompts'][ancora_latina].dropna()

        # Só executa o teste se houver dados em ambos os grupos
        if not inclusivo.empty and not neutro.empty:
            stat, p_value = mannwhitneyu(inclusivo, neutro, alternative='two-sided')
            
            st.markdown("### 📈 Relatório Estatístico")
            st.write(f"**P-Value:** `{p_value:.5f}` (Teste U de Mann-Whitney | Nível de Significância $\\alpha = 0.05$)")
            
            # Interpretação automática do P-value
            if p_value < 0.05:
                st.success("=> **Rejeita H0:** Existe diferença estatisticamente significativa na representação cultural.")
            else:
                st.error("=> **Aceita H0:** NÃO existe diferença significativa. O viés cultural eurocêntrico persiste mesmo com prompts inclusivos.")
        else:
            st.warning("Dados insuficientes para rodar o teste estatístico com os filtros selecionados.")
    else:
        st.error(f"A âncora '{ancora_latina}' não foi encontrada nos dados.")