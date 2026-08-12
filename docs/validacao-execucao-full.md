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

## Mann-Whitney no nivel da imagem

O teste compara as distribuicoes dos grupos de prompts sem duplicar observacoes
por rosto. Os p-values de Holm corrigem conjuntamente as duas hipoteses
primarias.

| Metrica | Mediana inclusiva | Mediana neutra | U | p | p de Holm | Decisao |
|---|---:|---:|---:|---:|---:|---|
| Margem de diversidade racial | 0,001182 | 0,000769 | 512 | 1,000000 | 1,000000 | Nao se rejeita H0 |
| Margem de diversidade de genero | 0,002347 | 0,003600 | 442 | 0,350723 | 0,701447 | Nao se rejeita H0 |

Esses resultados nao demonstram equivalencia entre prompts. Eles indicam que,
com este corpus e estas metricas, nao houve evidencia suficiente para rejeitar
as hipoteses nulas apos a correcao de Holm.

## Wilcoxon pareado

O teste de sensibilidade preserva os 32 pares definidos por modelo e prompt.

| Metrica | Diferenca mediana | W | p | p de Holm | Inclusiva maior | Neutra maior |
|---|---:|---:|---:|---:|---:|---:|
| Margem de diversidade racial | -0,000779 | 260 | 0,948544 | 0,948544 | 15 | 17 |
| Margem de diversidade de genero | -0,000879 | 223 | 0,454045 | 0,908090 | 15 | 17 |

Mann-Whitney e Wilcoxon convergem nesta execucao: nenhum dos dois fornece
evidencia estatistica para rejeitar H0 nas duas dimensoes analisadas.

## Dependencias cientificas restantes

1. Incorporar os CSVs e relatorios schema 2.0 ao repositorio e ao Streamlit.
2. Produzir o Ground Truth humano de deteccao e calcular IoU, precisao,
   revocacao e F1.
3. Integrar a baseline humana e calcular concordancia e correlacoes de
   Spearman com as margens CLIP.
4. Executar FairFace sobre os recortes do Ground Truth como pseudo-oraculo.
5. Atualizar Resultados e Conclusao sem apresentar ausencia de significancia
   como prova de igualdade.
