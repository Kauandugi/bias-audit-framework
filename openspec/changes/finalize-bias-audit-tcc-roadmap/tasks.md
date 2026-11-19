## 1. Especificação e contratos

- [x] 1.1 Substituir a especificação cultural pela análise multidimensional de diversidade.
- [x] 1.2 Definir schema 2.0, manifesto opcional e regras de ingestão recursiva.
- [x] 1.3 Preservar os contratos 1:N e reposicionar Ground Truth e FairFace.
- [x] 1.4 Reabrir texto, estatística e verificação dependentes dos novos escores.
- [x] 1.5 Especificar MLLM-as-a-Judge, referência humana perceptual e replicação do protocolo em corpus parceiro.

## 2. Pacote Python

- [x] 2.1 Criar `src/biasauditfw` com configuração, ingestão, contratos e pipeline.
- [x] 2.2 Implementar descoberta recursiva, Pillow, SHA-256 e duplicatas.
- [x] 2.3 Implementar DeepFace 1:N com zero, um ou vários rostos.
- [x] 2.4 Implementar protótipos CLIP, normalização L2 e duas margens.
- [x] 2.5 Implementar relatórios descritivos e inferenciais condicionais.
- [x] 2.6 Implementar validação COCO, IoU, bootstrap, baseline e FairFace.
- [x] 2.7 Implementar juízes opcionais, persistência rastreável e retomada idempotente.
- [x] 2.8 Implementar agregação humana multiavaliador e meta-avaliação dos juízes.
- [x] 2.9 Implementar comparação RetinaFace/MTCNN e adaptador WIDER FACE.
- [x] 2.10 Exportar a convergência da referência humana com CLIP e índices de Simpson, usando Spearman, bootstrap e Holm quando aplicável.
- [x] 2.11 Implementar adaptadores locais independentes para Qwen3-VL e Gemma 3, com revisão, quantização e geração rastreáveis.

## 3. Testes e manifesto

- [x] 3.1 Criar manifesto explícito para as 64 imagens originais.
- [x] 3.2 Testar diretórios rasos, profundos, Unicode, espaços e extensões.
- [x] 3.3 Testar manifesto ausente, parcial, duplicado e inconsistente.
- [x] 3.4 Testar imagem corrompida, conteúdo duplicado e link simbólico.
- [x] 3.5 Testar vetores CLIP, contrato 1:N e ausência de pseudorreplicação.
- [x] 3.6 Testar Mann–Whitney, Wilcoxon, Holm, Spearman, Simpson, IoU e concordância.
- [x] 3.7 Manter cobertura mínima de 85% nos módulos não neurais.
- [x] 3.8 Testar parsing, abstenção, retries, inversão A/B, retomada e agregação humana.
- [x] 3.9 Criar testes opt-in para Gemini, Groq e Qwen local.
- [x] 3.10 Implementar gate semântico para bloquear juízes com respostas degeneradas.
- [x] 3.11 Gerar manifesto schema 2.0 a partir de metadados laterais por imagem.
- [x] 3.12 Testar importação do Google Forms, mapa privado de 48 posições e cinco dimensões ordinais.
- [x] 3.13 Testar a convergência humana com CLIP e Simpson usando fixtures de resultado conhecido e casos com dados ausentes.
- [x] 3.14 Testar adaptadores locais com doubles determinísticos e testes reais opt-in separados por família de modelo.
- [x] 3.15 Reconhecer e testar imagens PNG/JPEG válidas com sufixo ausente ou incorreto, sem aceitar arquivos arbitrários.

## 4. Notebook e execução

- [x] 4.1 Transformar o notebook oficial em orquestrador do pacote.
- [x] 4.2 Adicionar preflight, testes, cache de modelos, smoke e full.
- [x] 4.3 Exportar schema 2.0 em diretório separado dos resultados legados.
- [x] 4.4 Criar validador independente que recalcula os testes.
- [x] 4.5 Executar smoke test real de quatro imagens no Colab T4.
- [x] 4.6 Executar as 64 imagens com o manifesto e validar 32 observações por grupo.
- [x] 4.7 Gerar o manifesto preliminar de 62 imagens reconhecidas por extensão e registrar duas inclusivas P4 como ausentes.
- [x] 4.8 Auditar os arquivos dados como ausentes e identificar duas imagens válidas com sufixo `._`, invalidando o delineamento preliminar de 62 imagens.
- [x] 4.9 Preservar a execução de 62 imagens apenas como diagnóstico superado, sem reutilizar seus resultados no texto final.
- [x] 4.10 Gerar notebook e snapshot determinísticos para pilotos locais independentes, mantendo o lote completo bloqueado sem amostra congelada.
- [x] 4.11 Regenerar o manifesto do corpus do Vitor com as 64 imagens válidas e executar novamente o pipeline e os validadores dependentes.

## 5. Validação empírica

- [x] 5.1 Manter pareamento IoU e métricas do detector.
- [x] 5.2 Executar RetinaFace no WIDER FACE e documentar o limite de domínio.
- [x] 5.3 Executar FairFace nos recortes RetinaFace e comparar atributos no corpus do Vitor; manter o corpus original pendente enquanto suas imagens-fonte não estiverem disponíveis localmente.
- [x] 5.4 Confirmar autorização do orientador e da coordenadora do Comitê de Ética para uso acadêmico dos agregados; manter respostas, perfis, consentimentos e mapa do formulário em armazenamento privado.
- [x] 5.5 Executar na T4 os pilotos semânticos independentes de Qwen3-VL e Gemma 3. Ambos foram bloqueados pelo gate: Qwen3 por consistência de ordem de gênero de 0,25 e Gemma por escores e escolhas constantes. O lote completo não foi liberado e os artefatos permanecem diagnósticos.
- [x] 5.6 Recalcular as comparações de detectores e FairFace após incorporar as duas imagens P4 válidas; reconstruir o mapa das 48 imagens e recalcular concordância, efeitos pareados e convergência humana contra o corpus de 64 imagens.

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
- [x] 7.5 Atualizar resultados e conclusão após a execução T4 dos dois corpora.
- [x] 7.6 Aplicar humanização depois da revisão técnica.
- [x] 7.7 Citar Gu et al. e documentar MLLM-as-a-Judge, vieses e meta-avaliação.
- [x] 7.8 Corrigir metodologia, resultados, resumo, abstract e conclusão para o corpus parceiro de 64 imagens e para a situação ética efetivamente documentada.

## 8. Verificação final

- [x] 8.1 Validar JSON do notebook e executar testes locais.
- [x] 8.2 Incorporar CSVs e relatórios schema 2.0 no repositório.
- [x] 8.3 Compilar o TCC e verificar consistência entre método, dados e conclusões.
- [x] 8.4 Alinhar código e documentação: Wilcoxon principal e Mann--Whitney como sensibilidade.
- [x] 8.5 Registrar o Parecer Consubstanciado final favorável n. 8.854.629, de 5 de novembro de 2026, CAAE 02077126.7.0000.8030, e a autorização específica informada pelo pesquisador para uso acadêmico agregado.
- [x] 8.6 Preservar respostas individuais e perfis em armazenamento privado, com metadados que liberam somente agregados por imagem.
- [x] 8.7 Incorporar alfa ordinal, testes humanos e convergência com CLIP e Simpson ao texto, dashboard e checkpoint, mantendo MLLMs reprovados fora das conclusões.
- [x] 8.8 Executar a regressão completa, validar o OpenSpec, compilar o PDF e auditar a consistência final dos artefatos admissíveis.
