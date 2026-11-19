# Validacao da execucao completa no Colab T4

Data da execucao: 12 de agosto de 2026 (UTC).

Revisao executada: `2ecaa2ccfffdbfae82714ec1642b054a133720f2`.

## Parecer

A execucao computacional do schema 2.0 foi aprovada. O notebook processou as 64
imagens do manifesto, registrou 413 rostos segundo o contrato 1:N e produziu os
relatorios de diversidade racial percebida e de apresentacao de genero no nivel
da imagem. O validador independente terminou com codigo de retorno zero.

Este parecer confirma reprodutibilidade e integridade computacional. Ele nao
comprova a acuracia do detector nem transforma as classificacoes do DeepFace ou
os escores do CLIP em identidades demograficas reais.

## Evidencias da execucao

| Verificacao | Resultado |
|---|---|
| Hardware | Tesla T4 com CUDA |
| Schema | `2.0` |
| Imagens validas | 64 |
| Grupos inferenciais | 32 inclusivas e 32 neutras |
| Rostos registrados | 413 |
| Testes deterministas no Colab | Aprovados (`returncode=0`) |
| Validador independente | Aprovado (`returncode=0`) |
| Erros no notebook exportado | 0 |
| Ground Truth | Desativado, aguardando anotacoes humanas |
| Baseline humana | Desativada nesta execucao |
| FairFace | Desativado nesta execucao |

Os artefatos cientificos foram gravados em:

`/content/drive/MyDrive/Estudo UNAL/BiasAuditFW outputs schema 2/full`

Os registros locais da execucao estao em:

`execucao/colab/full-20260812T024020Z/`

## Mann-Whitney no nivel da imagem (sensibilidade)

O teste compara as distribuicoes dos grupos de prompts sem duplicar observacoes
por rosto. Os p-values de Holm corrigem conjuntamente as duas dimensoes. Depois
da revisao metodologica, este teste permanece como sensibilidade; Wilcoxon e o
teste principal porque o corpus possui pares neutro-inclusivo.

| Metrica | Mediana inclusiva | Mediana neutra | U | p | p de Holm | Decisao |
|---|---:|---:|---:|---:|---:|---|
| Margem de diversidade racial | 0,001182 | 0,000769 | 512 | 1,000000 | 1,000000 | Nao se rejeita H0 |
| Margem de diversidade de genero | 0,002347 | 0,003600 | 442 | 0,350723 | 0,701447 | Nao se rejeita H0 |

Esses resultados nao demonstram equivalencia entre prompts. Eles indicam que,
com este corpus e estas metricas, nao houve evidencia suficiente para rejeitar
as hipoteses nulas apos a correcao de Holm.

## Wilcoxon pareado (analise principal)

O teste principal preserva os 32 pares definidos por modelo e prompt.

| Metrica | Diferenca mediana | W | p | p de Holm | Inclusiva maior | Neutra maior |
|---|---:|---:|---:|---:|---:|---:|
| Margem de diversidade racial | -0,000779 | 260 | 0,948544 | 0,948544 | 15 | 17 |
| Margem de diversidade de genero | -0,000879 | 223 | 0,454045 | 0,908090 | 15 | 17 |

No corpus original, Mann--Whitney e Wilcoxon não rejeitam $H_0$ nas duas
dimensões. Na replicação corrigida, ambos rejeitam $H_0$ para a margem de
gênero após Holm e não rejeitam para a margem racial. Wilcoxon permanece como
teste principal porque preserva o pareamento neutro-inclusivo.

## Estado das validações posteriores

1. RetinaFace no WIDER FACE: concluído e recalculado de forma independente.
2. RetinaFace--MTCNN e DeepFace--FairFace: concluídos no corpus do Vitor. As
   imagens-fonte do corpus original não estão disponíveis localmente para essa
   repetição.
3. Juízes multimodais: Gemini e Groq ficaram incompletos por limite de cota.
   Qwen2.5-VL, Qwen3-VL e Gemma 3 completaram pilotos locais, mas nenhum passou
   pelo gate semântico. A execução completa permaneceu bloqueada, e esses
   registros diagnósticos não entram nas conclusões empíricas.
4. Replicação do protocolo em corpus parceiro: concluída com 64 imagens e 32
   pares comparáveis. Duas imagens P4 antes dadas como ausentes eram arquivos
   válidos cujos nomes terminavam em `._`.
5. Referência humana perceptual: respostas individuais e perfis permanecem
   privados. Após autorização do orientador e da coordenadora do Comitê de
   Ética, agregados por imagem e relatórios derivados foram incorporados ao
   texto, ao dashboard e ao checkpoint.
6. Resultados e conclusão: atualizados sem tratar ausência de significância
   como prova de igualdade.
