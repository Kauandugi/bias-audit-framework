## MODIFIED Requirements

### Requirement: posicionamento metodológico

O texto SHALL classificar a pesquisa como aplicada, quantitativa, exploratória e
comparativa. O TCC SHALL usar “análise” em vez de alegar auditoria formal.

#### Scenario: classificação do trabalho
- **WHEN** resumo, introdução ou metodologia descrevem o delineamento
- **THEN** a redação usa pesquisa aplicada, quantitativa, exploratória e comparativa

### Requirement: novo alvo semântico

Resumo, abstract e capítulos SHALL tratar diversidade racial percebida e
diversidade de apresentação de gênero como alvos CLIP independentes. A adequação
cultural latino-americana SHALL permanecer apenas como antecedente histórico do
artigo publicado.

#### Scenario: descrição do CLIP
- **WHEN** o método semântico é explicado
- **THEN** o texto apresenta quatro famílias de protótipos, duas margens e normalização L2

### Requirement: níveis de evidência

Resultados SHALL separar cobertura, detector, pseudo-oráculo, composição por
face, CLIP por imagem, baseline humana e testes estatísticos.

#### Scenario: apresentação de uma métrica
- **WHEN** um resultado é informado
- **THEN** o texto identifica se a unidade é imagem, rosto ou caixa anotada

### Requirement: interpretação estatística

Para `p >= 0,05`, o texto MUST dizer “não se rejeita H0”. O Efeito Patchwork MUST
ser uma interpretação condicionada à convergência entre CLIP, baseline humana e
composição facial, não uma prova causal.

#### Scenario: resultado não significativo
- **WHEN** o p-value corrigido é maior ou igual a 0,05
- **THEN** o texto informa evidência insuficiente sem afirmar igualdade

### Requirement: limites das categorias

O texto MUST dizer que CLIP mede alinhamento semântico e que DeepFace/FairFace
classificam aparências percebidas em faces sintéticas. Nenhum deles infere
identidade, nacionalidade, ancestralidade, miscigenação ou sexo biológico.

#### Scenario: categoria automática
- **WHEN** uma categoria facial ou margem CLIP é discutida
- **THEN** ela é descrita como saída operacional, não como identidade real

### Requirement: humanização com rigor

A humanização SHALL ocorrer depois da revisão metodológica. Uma segunda leitura
MUST confirmar equações, unidades, incertezas, citações e correspondência com os
outputs schema 2.0.

#### Scenario: revisão final
- **WHEN** um capítulo é humanizado
- **THEN** uma revisão técnica posterior confirma que o significado não mudou
