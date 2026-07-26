## MODIFIED Requirements

### Requirement: duas unidades de análise

O sistema SHALL exportar `audit_images.csv` com uma linha por imagem e
`audit_faces.csv` com uma linha por rosto detectado. Imagens sem rosto MUST
permanecer na tabela de imagens com `num_faces = 0`.

#### Scenario: múltiplos rostos
- **WHEN** o detector retorna N rostos válidos
- **THEN** a imagem tem `num_faces = N` e a tabela facial contém N linhas relacionadas

### Requirement: identificadores determinísticos

`imagem_id` SHALL derivar de `dataset_id` e SHA-256 do conteúdo. `face_id` SHALL
derivar de `imagem_id` e da ordenação espacial defensiva das caixas.

#### Scenario: mudança de estrutura de pastas
- **WHEN** os mesmos bytes são movidos dentro do mesmo dataset
- **THEN** `imagem_id` permanece estável

### Requirement: separação do CLIP

Nenhuma coluna iniciada por `clip_` SHALL aparecer na tabela facial. Estatística
semântica MUST usar somente imagens únicas.

#### Scenario: proteção contra pseudorreplicação
- **WHEN** uma tabela contém `imagem_id` duplicado ou conteúdo duplicado
- **THEN** a análise inferencial falha explicitamente

