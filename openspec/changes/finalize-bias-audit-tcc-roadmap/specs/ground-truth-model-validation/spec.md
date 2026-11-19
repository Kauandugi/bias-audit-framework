## ADDED Requirements

### Requirement: benchmark público da detecção

O sistema SHALL avaliar RetinaFace em caixas anotadas do WIDER FACE. O texto
MUST descrever esse resultado como consistência em benchmark público e MUST NOT
generalizá-lo como acurácia no corpus sintético.

#### Scenario: associação segura
- **WHEN** anotações WIDER FACE são importadas
- **THEN** cada imagem é associada por identificador e caminho relativo, nunca apenas pelo nome

### Requirement: avaliação do detector

Predições e caixas de benchmark SHALL ser pareadas um-para-um com `IoU >= 0,5`.
O relatório MUST conter precisão, revocação, F1, AP50, IoU médio, falsos
positivos e falsos negativos.

#### Scenario: avaliação completa
- **WHEN** anotações de benchmark e predições estão disponíveis
- **THEN** o relatório geral e os estratos são exportados mesmo com desempenho baixo

### Requirement: pseudo-referência de detecção sintética

RetinaFace e MTCNN SHALL ser executados sobre as mesmas imagens sintéticas. A
comparação SHALL reportar diferença de contagem, IoU das caixas pareadas e taxa
de detecções não correspondidas. O relatório MUST NOT chamar essa comparação de
acurácia ou Ground Truth.

#### Scenario: discordância entre detectores
- **WHEN** os detectores retornam contagens ou caixas diferentes
- **THEN** a divergência é preservada como resultado e nenhum detector é promovido a verdade

### Requirement: referência humana perceptual

O sistema SHALL aceitar, para 48 posições mapeadas por `imagem_id`, notas de 1 a
5 de múltiplos avaliadores para diversidade racial, equilíbrio de apresentação
de gênero, contexto latino-americano, diversidade corporal e qualidade visual.
As duas primeiras dimensões SHALL ser primárias e as demais SHALL permanecer
exploratórias. O relatório MUST conter alfa de Krippendorff ordinal, mediana e
intervalo interquartil para as cinco dimensões, além da correlação de Spearman
das duas dimensões primárias com a margem automática correspondente. Respostas
individuais e perfil dos participantes MUST permanecer fora dos artefatos
públicos.

#### Scenario: baseline correspondente
- **WHEN** avaliações humanas agregadas e escores CLIP compartilham `imagem_id`
- **THEN** cada dimensão humana é comparada somente à margem correspondente

#### Scenario: exportação privada do Google Forms
- **WHEN** o CSV bruto do formulário é importado
- **THEN** somente as cinco notas e códigos locais de participante entram na tabela longa, enquanto carimbo, conta e perfil são descartados

### Requirement: admissibilidade ética da referência humana

Resultados de participantes SHALL entrar nas conclusões e nos artefatos públicos
somente quando houver parecer ético final favorável ou autorização institucional
explícita aplicável ao uso agregado. A base da autorização MUST ser registrada
sem converter orientação limitada a agregados em aprovação irrestrita da coleta.

#### Scenario: parecer pendente
- **WHEN** o documento disponível informa situação `Pendente`
- **THEN** respostas individuais permanecem privadas e agregados são excluídos até parecer favorável ou autorização institucional explícita aplicável ao seu uso

#### Scenario: uso agregado autorizado
- **WHEN** a autorização institucional permite usar resultados agregados sem identificadores
- **THEN** somente agregados por `imagem_id` e relatórios derivados entram nas conclusões e no pacote público

#### Scenario: parecer final favorável
- **WHEN** o parecer final favorável e a autorização específica para uso agregado estão registrados em armazenamento privado
- **THEN** os agregados podem entrar no texto e no pacote público, enquanto respostas individuais, perfis, consentimentos e mapas privados continuam excluídos

### Requirement: FairFace como pseudo-oráculo

FairFace SHALL operar nos recortes produzidos pelo RetinaFace. A comparação com DeepFace
MUST reportar concordância, kappa, macro-F1 e matriz de confusão após
harmonização, sem tratar FairFace como verdade demográfica ou como validação do
detector.

#### Scenario: comparação de atributos
- **WHEN** uma categoria não possui mapeamento explícito
- **THEN** ela permanece não harmonizável e não é forçada a outra classe

### Requirement: MLLM-as-a-Judge rastreável

O sistema SHALL produzir notas individuais e escolhas pareadas por pelo menos
duas famílias independentes de juízes multimodais locais configuradas
explicitamente. Provedores em nuvem MAY complementar a análise, mas MUST NOT ser
uma dependência do pipeline ou do cronograma. Cada registro MUST conter modelo,
versão, prompt, repetição, ordem e hash da entrada. O pipeline principal MUST
continuar funcionando sem chaves de MLLM.

#### Scenario: modelo indisponível
- **WHEN** o provedor não oferece o identificador configurado
- **THEN** a execução falha explicitamente e não escolhe outro modelo

#### Scenario: inversão de posição
- **WHEN** um par é avaliado como A/B e B/A
- **THEN** as escolhas são normalizadas para os mesmos `imagem_id` antes de medir consistência

#### Scenario: execução local sem créditos de API
- **WHEN** as credenciais de Gemini e Groq estão ausentes
- **THEN** Qwen3-VL e Gemma 3 podem ser avaliados localmente e de forma independente

### Requirement: gate semântico antes da execução completa

Um juiz local MUST passar por um piloto determinístico antes de avaliar todo o
corpus. O gate MUST verificar cobertura pointwise e pairwise, taxa de abstenção,
variação das notas, taxa de `not_evaluable`, variação das respostas pareadas e
consistência após inversão A/B. Uma falha MUST interromper a execução completa e
preservar os artefatos como diagnóstico não utilizável.

#### Scenario: respostas constantes
- **WHEN** o piloto produz abstinência excessiva ou respostas sem variação
- **THEN** o relatório marca `passed=false` e o lote completo não é iniciado

#### Scenario: gate aprovado
- **WHEN** todas as verificações operacionais são satisfeitas
- **THEN** o modelo pode seguir para o corpus, sem que isso seja descrito como acurácia

#### Scenario: amostra completa não congelada
- **WHEN** uma execução completa é habilitada sem o CSV de amostra humana do corpus
- **THEN** o processo falha antes de enviar qualquer imagem ao juiz

### Requirement: convergência facial secundária

Índices de Simpson normalizados SHALL ser calculados sobre categorias DeepFace e
FairFace por imagem. Imagens com menos de dois rostos MUST receber valor ausente.

#### Scenario: imagem com um rosto
- **WHEN** uma imagem tem menos de dois rostos válidos
- **THEN** o índice de Simpson é ausente, não zero
