# BiasAuditFW

Framework de pesquisa para análise quantitativa e exploratória de vieses de
representação em imagens geradas por IA. O projeto é o Trabalho de Conclusão de
Curso em Ciência da Computação na UEMS e continua uma pesquisa realizada durante
intercâmbio na Universidad Nacional de Colombia.

## Pergunta de pesquisa

Como imagens produzidas por modelos generativos variam entre prompts neutros e
inclusivos quanto à diversidade racial percebida e à diversidade de apresentação
de gênero?

O sistema analisa aparências em conteúdo sintético. Ele não infere identidade,
ancestralidade, nacionalidade, miscigenação, sexo biológico ou autenticidade
cultural.

## Arquitetura

O notebook oficial orquestra o pacote `src/biasauditfw`:

1. A ingestão encontra imagens recursivamente e valida conteúdo, duplicatas e
   manifesto.
2. DeepFace com RetinaFace detecta zero, um ou vários rostos.
3. CLIP calcula duas margens independentes no nível da imagem.
4. A estatística usa somente imagens únicas e explicitamente elegíveis.
5. O WIDER FACE e os modelos MTCNN/FairFace produzem verificações
   complementares; as avaliações humanas entram apenas como agregados
   autorizados.

```text
racial_margin = cos(image, racial_diversity) - cos(image, racial_homogeneity)
gender_margin = cos(image, gender_diversity) - cos(image, gender_homogeneity)
```

Cada polo CLIP combina três frases. Os vetores das frases são normalizados por
L2, promediados e normalizados novamente. A imagem também é normalizada por L2.
As margens medem alinhamento semântico, não contagem ou proporção de pessoas.

## Schema 2.0

| Artefato | Unidade | Uso |
| --- | --- | --- |
| `audit_images.csv` | uma linha por imagem | CLIP, filtros e estatística |
| `audit_faces.csv` | uma linha por rosto | caixas e categorias DeepFace |
| `ingestion_report.csv` | uma linha por candidato | validade e duplicatas |
| `run_metadata.json` | uma linha por execução | versões e configuração |
| `inference_status.json` | uma linha por execução | elegibilidade estatística |

Os outputs anteriores permanecem como legado e não devem sustentar as novas
conclusões.

Os artefatos da execução completa validada do corpus original estão em
[`data/schema2/`](data/schema2/). O CSV anterior permanece em `data/` apenas para
compatibilidade histórica.

## Datasets e manifesto

A ingestão suporta `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif` e `.tiff`
em qualquer profundidade. Arquivos com sufixo ausente ou incorreto também são
aceitos quando o Pillow confirma seu conteúdo como imagem válida. A estrutura
das pastas não é interpretada.

O manifesto é opcional. Quando usado, `relative_path` é obrigatório e estas
colunas são aceitas:

```text
source_image_id, modelo_ia, tipo_prompt, prompt_id, pair_id, analysis_eligible
```

O corpus original usa [data/original_64_manifest.csv](data/original_64_manifest.csv).
Sem manifesto, as imagens são processadas e recebem estatística descritiva, mas a
inferência neutro/inclusivo é ignorada com motivo explícito.

Datasets que possuem um `metadata.json` ao lado de cada imagem podem gerar o
manifesto sem depender da estrutura de diretórios:

```bash
python scripts/build_sidecar_manifest.py DATASET_ROOT manifest.csv \
  --dataset-id replication --expected-images 16
```

O corpus do Vitor usa um layout legado misto e possui um construtor dedicado:

```bash
python scripts/build_victor_manifest.py \
  "Novo Dataset de imagens do Vitor 2/Geração de imagens" \
  data/victor_replication_manifest.csv \
  --dataset-id vitor-64 --expected-images 64
```

Depois do manifesto, o mesmo pipeline do notebook pode ser executado por CLI:

```powershell
uv run --python 3.12 --extra neural --frozen python scripts/run_corpus.py `
  "Novo Dataset de imagens do Vitor 2/Geração de imagens" `
  data/victor_replication_manifest.csv `
  output/runs/vitor-64 `
  --dataset-id vitor-64 `
  --expected-images 64
```

O manifesto corrigido registra 64 imagens em 32 pares completos. As versões
inclusivas P4 de Midjourney e Whisk estavam presentes, mas seus nomes terminavam
em `._`; a descoberta anterior baseada somente em extensão não as reconheceu.
Os arquivos foram validados como PNG e JPEG pelo conteúdo, sem reconstrução ou
substituição.

## Google Colab com T4

1. Envie ou abra `Framework_Auditoria_Viés_IA_Generativa.ipynb` no Colab.
2. Selecione uma GPU T4.
3. Confirme `DATASET_ROOT`, `OUTPUT_ROOT` e `MANIFEST_PATH`.
4. Execute de cima para baixo com `RUN_FULL = False`.
5. Valide o smoke test de quatro imagens.
6. Altere `RUN_FULL = True` e execute novamente a partir da configuração.
7. Rode o validador final com `--expected-images 64 --require-cuda --require-inference`.

O notebook grava `smoke/` e `full/` em um diretório schema 2.0 separado.

### Execução automatizada pelo Colab CLI


No Windows, o CLI oficial é executado pelo Ubuntu 24.04 no WSL. Depois da
instalação e autenticação descritas em
[docs/colab-cli-workflow.md](docs/colab-cli-workflow.md), use:

```powershell
.\tools\colab.ps1 doctor
.\tools\colab.ps1 repair-cli
.\tools\colab.ps1 auth
.\tools\colab.ps1 gpu-probe
.\tools\colab.ps1 start
.\tools\colab.ps1 mount-drive
.\tools\colab.ps1 smoke
```

Substitua `smoke` por `full` somente após validar a execução reduzida. O wrapper
registra notebook e logs em `execucao/colab/` e encerra a sessão ao terminar ou
quando ocorre uma falha.

## Testes locais

```bash
python -m pip install -e ".[test]"
python -m pytest -q --cov=biasauditfw --cov-report=term-missing
```

A meta é cobertura mínima de 85% nos módulos não neurais. Os testes cobrem
ingestão arbitrária, manifesto, 1:N, vetores CLIP, pseudorreplicação,
Mann–Whitney, Wilcoxon, Holm, Spearman com bootstrap, Simpson, IoU,
consentimento e concordância ordinal.

Depois de congelar código, texto e resultados, gere o manifesto de hashes:

```bash
python scripts/build_tcc_release_checkpoint.py
```

O comando falha se o repositório estiver sujo. Durante a preparação, use
`--allow-dirty` apenas para criar um candidato marcado como não final. O
checkpoint inclui somente artefatos públicos; respostas individuais, perfis,
consentimentos e o mapa privado do formulário são excluídos.

## Estatística

Wilcoxon pareado é o teste principal para as diferenças neutro-inclusivo por
`pair_id`. Mann–Whitney compara as distribuições sem usar o pareamento e funciona
como análise de sensibilidade. A correção de Holm cobre as duas dimensões
primárias e, em família separada, as análises exploratórias por modelo.

Um `p >= 0,05` é descrito como evidência insuficiente para rejeitar H0, não como
prova de igualdade.

## Validação

- RetinaFace: conjunto público de referência WIDER FACE e comparação de consistência com
  MTCNN no domínio sintético, sem chamar o segundo detector de Ground Truth.
- Avaliação humana perceptual: 70 participantes com consentimento válido
  avaliaram 48 imagens, totalizando 3.360 avaliações participante--imagem e
  16.800 notas nas cinco dimensões. Respostas individuais, perfis, consentimentos e o mapa
  privado não são publicados. Somente agregados por `imagem_id` e relatórios
  derivados entram no dashboard, no texto e no checkpoint.
- FairFace: segunda referência algorítmica aplicada aos mesmos recortes
  RetinaFace, com concordância, kappa, macro-F1 e matriz de confusão.
- MLLM-as-a-Judge: etapa opcional e separada, com critério de liberação semântica, rastreabilidade,
  repetições e controle de ordem.
  No piloto canônico de oito imagens, Gemini e Groq ficaram incompletos por
  limites de cota, e o Qwen2.5-VL local não passou pelo critério. Há adaptadores
  e registros diagnósticos separados para Qwen3-VL e Gemma 3, mas eles não são
  combinados com esse piloto nem usados como evidência do estudo. A execução
  completa não foi liberada.
- Simpson: medida convergente secundária sobre categorias faciais; fica ausente
  quando há menos de dois rostos.

FairFace é uma referência secundária, não uma verdade demográfica.

O protocolo humano vigente está em
[docs/protocolo-validacao-perceptual-unesp.md](docs/protocolo-validacao-perceptual-unesp.md).
O projeto e o instrumento receberam parecer final favorável: Parecer
Consubstanciado nº 8.854.629, de 5 de novembro de 2026, CAAE
02077126.7.0000.8030. Esse parecer é a referência ética adotada para o formulário
aplicado na UNESP, conforme orientação acadêmica recebida pelo pesquisador. A
coleta ocorreu entre 9 e 15 de outubro de 2026, antes da emissão do parecer, e
essa cronologia permanece registrada nos metadados. A autorização específica
informada para uso acadêmico agregado permanece em arquivo privado. O script
`scripts/prepare_unesp_human_ratings.py` exclui recusas de consentimento,
descarta perfil e identificadores e grava somente artefatos publicáveis em
`data/human-unesp-aggregate/`.

No conjunto de validação do WIDER FACE, o RetinaFace obteve precisão de 0,9762,
revocação de 0,5675, F1 de 0,7177 e AP50 global recalculado pelo framework de
0,5642. Esse conjunto de referência não mede
acurácia no domínio sintético.

### Resultados consolidados

- As 128 imagens formam 64 pares neutro--inclusivo e contêm 998 rostos
  detectados.
- Nas avaliações humanas, as duas dimensões primárias não apresentaram
  diferença significativa após Holm. Contexto latino-americano e diversidade
  corporal foram maiores nas imagens inclusivas dos dois conjuntos.
- As margens CLIP não apresentaram associação significativa com as notas
  humanas.
- As correlações simples entre a percepção humana e o índice de Gini--Simpson
  não permaneceram significativas após Holm. Na replicação, a correlação racial
  parcial foi positiva após controlar o número de rostos, mas é tratada como
  exploratória porque envolve 23 imagens e não foi acompanhada pela correlação
  simples correspondente.
- Nenhum piloto MLLM atingiu os critérios de cobertura e estabilidade; por isso,
  essas execuções não sustentam as conclusões do TCC.

## Dashboard

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

O Streamlit carrega os artefatos validados de `data/schema2/`, filtra as duas
tabelas pelos mesmos `imagem_id`, separa KPIs de imagens e rostos, mostra as
duas margens e só apresenta agregados humanos quando os metadados registram
explicitamente que eles são admissíveis como evidência final.

## Estrutura

```text
src/biasauditfw/                               # pacote testável
Framework_Auditoria_Viés_IA_Generativa.ipynb  # orquestrador Colab/T4
tests/                                         # testes e validador de outputs
data/original_64_manifest.csv                  # manifesto do corpus original
data/schema2/                                  # outputs validados do schema 2.0
data/schema2-vitor/                            # replicação de 64 imagens
data/human-unesp-aggregate/                    # agregados humanos autorizados
output/validation/                             # validações técnicas publicáveis
app.py                                         # dashboard Streamlit
openspec/                                      # especificação e tarefas
notebooks/bias_audit_pipeline.ipynb            # referência histórica
```

## Pesquisa anterior

O corpus e a baseline humana derivam de *Exploring Bias in AI-Generated
Imagery: A Design-Research Case Study in University Visual Communication*
(INTED 2026): <https://doi.org/10.21125/inted.2026.2311>.
