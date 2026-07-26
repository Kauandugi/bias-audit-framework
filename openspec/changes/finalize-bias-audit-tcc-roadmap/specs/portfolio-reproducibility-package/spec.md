## MODIFIED Requirements

### Requirement: notebook oficial reproduzível

O notebook da raiz SHALL instalar `src/biasauditfw`, montar o Drive, executar
preflight e testes, carregar modelos uma vez e separar smoke/full.

#### Scenario: sessão limpa do Colab
- **WHEN** o notebook é executado de cima para baixo
- **THEN** o pacote é instalado antes de qualquer importação do pipeline

### Requirement: execução genérica

O README SHALL documentar o perfil original de 64 imagens e o perfil genérico
sem manifesto. Outputs schema 2.0 MUST ficar separados dos artefatos legados.

#### Scenario: segundo dataset
- **WHEN** um usuário configura manifesto ausente e contagem esperada ausente
- **THEN** a documentação explica que apenas descritivas serão produzidas

### Requirement: dashboard consistente

O Streamlit SHALL carregar tabelas de imagem e rosto, aplicar filtros pelas
chaves `imagem_id` e exibir as duas margens de diversidade com suas limitações.

#### Scenario: filtro por modelo
- **WHEN** um modelo gerador é selecionado
- **THEN** rostos e imagens exibidos pertencem aos mesmos identificadores

### Requirement: aceite automatizado

Pytest SHALL manter cobertura mínima de 85% nos módulos não neurais. Um validador
independente MUST verificar o schema e recalcular Mann–Whitney e Wilcoxon a partir
dos CSVs exportados.

#### Scenario: output alterado
- **WHEN** um relatório estatístico diverge da recomputação
- **THEN** o validador encerra com erro explícito
