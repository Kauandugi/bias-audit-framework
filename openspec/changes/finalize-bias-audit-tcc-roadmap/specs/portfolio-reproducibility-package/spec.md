## ADDED Requirements

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

### Requirement: validação opcional e separada

MLLM-as-a-Judge MUST ser uma etapa opcional, executável após os CSVs canônicos.
A ausência de chaves de API MUST NOT impedir ingestão, CLIP, faces, estatística
principal, dashboard ou os juízes locais explicitamente configurados. Artefatos
de juiz MUST ficar em diretório próprio.

#### Scenario: execução sem credenciais
- **WHEN** `GEMINI_API_KEY` e `GROQ_API_KEY` estão ausentes
- **THEN** o pipeline principal funciona e os provedores em nuvem são ignorados sem impedir os pilotos locais
