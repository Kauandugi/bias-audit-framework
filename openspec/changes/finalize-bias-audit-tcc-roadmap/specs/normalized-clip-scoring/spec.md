## MODIFIED Requirements

### Requirement: protótipos textuais por dimensão

O sistema SHALL representar diversidade e homogeneidade racial percebida e
diversidade e homogeneidade de apresentação de gênero. Cada polo MUST conter
múltiplas frases versionadas.

#### Scenario: carregamento do CLIP
- **WHEN** o modelo é inicializado
- **THEN** todas as frases são codificadas uma vez e os quatro protótipos ficam em cache

### Requirement: cosseno L2 explícito

Cada vetor textual SHALL ser normalizado por L2 antes da média. Cada protótipo
médio e cada vetor de imagem MUST ser normalizado novamente antes do produto
escalar.

#### Scenario: faixa numérica
- **WHEN** uma imagem é processada
- **THEN** cada cosseno pertence a `[-1,1]` e cada margem a `[-2,2]`

### Requirement: margens independentes

O sistema SHALL exportar `racial_diversity - racial_homogeneity` e
`gender_diversity - gender_homogeneity`. O sistema MUST NOT aplicar softmax,
contagem ou combinar as duas margens em um escore único.

#### Scenario: exportação
- **WHEN** o CLIP conclui uma imagem
- **THEN** seis colunas são gravadas somente em `audit_images.csv`

### Requirement: interpretação limitada

Os escores SHALL ser descritos como alinhamento semântico com diversidade
percebida, não como identidade ou proporção demográfica.

#### Scenario: leitura de um escore
- **WHEN** uma margem é exibida no notebook, dashboard ou texto
- **THEN** sua descrição exclui contagem, identidade e proporção de pessoas
