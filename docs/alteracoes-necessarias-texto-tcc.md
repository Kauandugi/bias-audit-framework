# Alterações necessárias no texto do TCC

Este inventário acompanha a migração para o schema 2.0 e separa o que já foi
reescrito do que depende dos resultados do Google Colab.

## Terminologia geral

- Manter o título atual sobre análise exploratória.
- Descrever a pesquisa como aplicada, quantitativa, exploratória e comparativa.
- Usar “análise” nas alegações científicas. `BiasAuditFW` permanece como nome do
  projeto.
- Evitar “prova”, “comprova matematicamente”, “verdade demográfica” e “aceita
  H0”.
- Usar “não se rejeita H0” quando `p >= 0,05`.
- Tratar diversidade como aparência percebida, não identidade.

## Resumo e abstract

- [x] Remover a oposição latino-americano/não latino-americano.
- [x] Apresentar as margens racial e de apresentação de gênero.
- [x] Distinguir imagem e rosto como unidades.
- [ ] Inserir resultados somente após a execução completa schema 2.0.

## Introdução

- [x] Citar `HENRICKSILVA2026EXP` como origem do corpus e da baseline.
- [x] Explicar por que “latino-americano” não é uma categoria facial adequada.
- [x] Reformular pergunta, hipóteses e objetivos para duas dimensões.
- [x] Restringir o Efeito Patchwork à convergência das evidências.

## Trabalhos relacionados

- [x] Manter DeepFace, FairFace e IBGE com limites claros.
- [x] Explicar agregação de frases no CLIP.
- [x] Citar a limitação de contagem do CLIP (`paiss2023counting`).
- [x] Reposicionar a adequação cultural como antecedente do artigo.

## Metodologia

- [x] Documentar pacote Python, notebook orquestrador e Colab/T4.
- [x] Descrever ingestão recursiva, manifesto, Pillow, SHA-256 e duplicatas.
- [x] Atualizar o identificador da imagem para dataset mais hash do conteúdo.
- [x] Preservar DeepFace 1:N e ausência de linha facial artificial.
- [x] Inserir quatro famílias de protótipos e duas equações de margem.
- [x] Registrar normalização L2 antes e depois da média textual.
- [x] Explicar que CLIP não usa softmax, contagem ou escore combinado.
- [x] Manter Ground Truth, baseline humana, FairFace e Simpson.
- [x] Aplicar Mann–Whitney e Wilcoxon separadamente, com Holm.

## Resultados e discussão

- [x] Substituir a estrutura cultural pelas duas dimensões de diversidade.
- [ ] Inserir cobertura, duplicatas e erros da nova ingestão.
- [ ] Inserir precisão, revocação, F1, IoU, FP, FN e bootstrap.
- [ ] Inserir kappa humano, Spearman e concordância DeepFace–FairFace.
- [ ] Inserir descritivas e gráficos das duas margens.
- [ ] Inserir U, Wilcoxon, tamanhos de efeito e correções de Holm.
- [ ] Discutir o Efeito Patchwork somente se houver convergência.

## Conclusão

- [x] Atualizar a contribuição técnica para schema 2.0 e datasets arbitrários.
- [x] Remover qualquer conclusão baseada no eixo cultural antigo.
- [ ] Finalizar a conclusão empírica após os resultados T4 e Ground Truth.

## Revisão de estilo

Foi aplicada uma primeira humanização depois da correção metodológica. A leitura
final ainda deverá conferir números, equações, citações e correspondência literal
com os CSVs. O texto deve permanecer direto, sem tom promocional ou conclusões
maiores que a evidência.

## Dependências externas

- Smoke test e execução completa no Colab T4.
- Duas anotações independentes de caixas faciais e consenso.
- Execução da baseline humana e do FairFace com os outputs schema 2.0.
- Incorporação dos CSVs, relatórios e gráficos finais.

