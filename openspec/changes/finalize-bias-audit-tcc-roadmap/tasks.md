## 1. Especificação e contratos

- [x] 1.1 Substituir a especificação cultural pela análise multidimensional de diversidade.
- [x] 1.2 Definir schema 2.0, manifesto opcional e regras de ingestão recursiva.
- [x] 1.3 Preservar os contratos 1:N, Ground Truth e FairFace válidos.
- [x] 1.4 Reabrir texto, estatística e verificação dependentes dos novos escores.

## 2. Pacote Python

- [x] 2.1 Criar `src/biasauditfw` com configuração, ingestão, contratos e pipeline.
- [x] 2.2 Implementar descoberta recursiva, Pillow, SHA-256 e duplicatas.
- [x] 2.3 Implementar DeepFace 1:N com zero, um ou vários rostos.
- [x] 2.4 Implementar protótipos CLIP, normalização L2 e duas margens.
- [x] 2.5 Implementar relatórios descritivos e inferenciais condicionais.
- [x] 2.6 Implementar validação COCO, IoU, bootstrap, baseline e FairFace.

## 3. Testes e manifesto

- [x] 3.1 Criar manifesto explícito para as 64 imagens originais.
- [x] 3.2 Testar diretórios rasos, profundos, Unicode, espaços e extensões.
- [x] 3.3 Testar manifesto ausente, parcial, duplicado e inconsistente.
- [x] 3.4 Testar imagem corrompida, conteúdo duplicado e link simbólico.
- [x] 3.5 Testar vetores CLIP, contrato 1:N e ausência de pseudorreplicação.
- [x] 3.6 Testar Mann–Whitney, Wilcoxon, Holm, Spearman, Simpson, IoU e concordância.
- [x] 3.7 Manter cobertura mínima de 85% nos módulos não neurais.

## 4. Notebook e execução

- [x] 4.1 Transformar o notebook oficial em orquestrador do pacote.
- [x] 4.2 Adicionar preflight, testes, cache de modelos, smoke e full.
- [x] 4.3 Exportar schema 2.0 em diretório separado dos resultados legados.
- [x] 4.4 Criar validador independente que recalcula os testes.
- [ ] 4.5 Executar smoke test real de quatro imagens no Colab T4.
- [ ] 4.6 Executar as 64 imagens com o manifesto e validar 32 observações por grupo.
- [ ] 4.7 Executar um segundo dataset sem pressupor estrutura de pastas.

## 5. Validação empírica

- [x] 5.1 Manter importação COCO e métricas do detector.
- [x] 5.2 Manter baseline humana com kappa e Spearman.
- [x] 5.3 Manter FairFace sobre recortes do Ground Truth.
- [ ] 5.4 Realizar duas anotações humanas e resolver o consenso.
- [ ] 5.5 Executar detector, baseline e pseudo-oráculo sobre os outputs schema 2.0.

## 6. Dashboard e documentação

- [x] 6.1 Atualizar Streamlit para as duas margens de diversidade.
- [x] 6.2 Filtrar imagens e rostos pelos mesmos `imagem_id`.
- [x] 6.3 Separar KPIs de imagens, rostos e cobertura.
- [x] 6.4 Atualizar README, instruções Colab e descrição dos artefatos.

## 7. Texto acadêmico

- [x] 7.1 Remover o antigo eixo cultural do resumo, abstract e capítulos.
- [x] 7.2 Inserir as duas equações, famílias de protótipos e limites do CLIP.
- [x] 7.3 Reposicionar a dimensão cultural como antecedente do artigo publicado.
- [x] 7.4 Reformular o Efeito Patchwork como interpretação condicionada à convergência.
- [ ] 7.5 Atualizar resultados e conclusão somente após a nova execução T4.
- [x] 7.6 Aplicar humanização depois da revisão técnica.

## 8. Verificação final

- [x] 8.1 Validar JSON do notebook e executar testes locais.
- [ ] 8.2 Incorporar CSVs e relatórios schema 2.0 no repositório.
- [ ] 8.3 Compilar o TCC e verificar consistência entre método, dados e conclusões.
