## MODIFIED Requirements

### Requirement: Ground Truth humano da detecção

Duas pessoas SHALL anotar caixas faciais independentemente em COCO. Divergências
de contagem ou caixas com IoU inferior a 0,5 SHALL passar por consenso.

#### Scenario: associação segura
- **WHEN** o COCO é importado
- **THEN** cada imagem é associada por caminho relativo ou `source_image_id`, nunca apenas pelo nome

### Requirement: avaliação do detector

Predições e caixas humanas SHALL ser pareadas um-para-um com `IoU >= 0,5`. O
relatório MUST conter precisão, revocação, F1, IoU médio, falsos positivos,
falsos negativos e intervalos bootstrap.

#### Scenario: avaliação completa
- **WHEN** Ground Truth e predições estão disponíveis
- **THEN** o relatório geral e os estratos são exportados mesmo com desempenho baixo

### Requirement: baseline humana

O sistema SHALL reutilizar `Racial_Diversity` para validar a margem racial e
`Gender_Representation` para validar a margem de apresentação de gênero. O
relatório MUST conter kappa ponderado entre R1/R2 e correlação de Spearman.

#### Scenario: baseline correspondente
- **WHEN** avaliações humanas e escores CLIP compartilham `source_image_id`
- **THEN** cada dimensão humana é comparada somente à margem correspondente

### Requirement: FairFace como pseudo-oráculo

FairFace SHALL operar nos recortes do Ground Truth. A comparação com DeepFace
MUST reportar concordância, kappa, macro-F1 e matriz de confusão após
harmonização, sem tratar FairFace como verdade demográfica.

#### Scenario: comparação de atributos
- **WHEN** uma categoria não possui mapeamento explícito
- **THEN** ela permanece não harmonizável e não é forçada a outra classe

### Requirement: convergência facial secundária

Índices de Simpson normalizados SHALL ser calculados sobre categorias DeepFace e
FairFace por imagem. Imagens com menos de dois rostos MUST receber valor ausente.

#### Scenario: imagem com um rosto
- **WHEN** uma imagem tem menos de dois rostos válidos
- **THEN** o índice de Simpson é ausente, não zero
