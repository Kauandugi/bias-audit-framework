## ADDED Requirements

### Requirement: descoberta recursiva independente da estrutura

O sistema SHALL buscar imagens em qualquer profundidade e suportar PNG, JPEG,
WEBP, BMP e TIFF sem diferenciar maiúsculas de minúsculas. O sistema MUST NOT
inferir modelo ou prompt pela posição da pasta.

#### Scenario: dataset sem manifesto
- **WHEN** imagens válidas existem em uma estrutura arbitrária
- **THEN** todas são processadas e os metadados analíticos permanecem ausentes

### Requirement: integridade da ingestão

O sistema SHALL validar cada imagem com Pillow, ignorar links simbólicos e o
diretório de saída, calcular SHA-256 e rejeitar conteúdo duplicado por padrão.

#### Scenario: duplicata por conteúdo
- **WHEN** dois caminhos contêm os mesmos bytes
- **THEN** a ingestão falha ou preserva explicitamente apenas o primeiro segundo a configuração

### Requirement: manifesto opcional

O manifesto MUST ter `relative_path` e MAY fornecer `source_image_id`,
`modelo_ia`, `tipo_prompt`, `prompt_id`, `pair_id` e `analysis_eligible`.

#### Scenario: inferência sem metadados
- **WHEN** grupos e pares válidos não estão disponíveis
- **THEN** o sistema exporta descritivas e registra o motivo da inferência ignorada

