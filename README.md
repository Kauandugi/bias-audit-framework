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
5. Ground Truth, baseline humana e FairFace produzem validações complementares.

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
em qualquer profundidade. A estrutura das pastas não é interpretada.

O manifesto é opcional. Quando usado, `relative_path` é obrigatório e estas
colunas são aceitas:

```text
source_image_id, modelo_ia, tipo_prompt, prompt_id, pair_id, analysis_eligible
```

O corpus original usa [data/original_64_manifest.csv](data/original_64_manifest.csv).
Sem manifesto, as imagens são processadas e recebem estatística descritiva, mas a
inferência neutro/inclusivo é ignorada com motivo explícito.

## Google Colab com T4

1. Envie ou abra `Framework_Auditoria_Viés_IA_Generativa.ipynb` no Colab.
2. Selecione uma GPU T4.
3. Confirme `DATASET_ROOT`, `OUTPUT_ROOT` e `MANIFEST_PATH`.
4. Execute de cima para baixo com `RUN_FULL = False`.
5. Valide o smoke test de quatro imagens.
6. Altere `RUN_FULL = True` e execute novamente a partir da configuração.
7. Rode o validador final com `--expected-images 64 --require-cuda`.

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
Mann–Whitney, Wilcoxon, Holm, Spearman, Simpson, IoU e concordância.

## Estatística

Mann–Whitney compara as distribuições neutra e inclusiva separadamente para as
duas margens. Wilcoxon pareado funciona como análise de sensibilidade por
`pair_id`. A correção de Holm cobre as duas hipóteses primárias e, em família
separada, as análises por modelo.

Um `p >= 0,05` é descrito como evidência insuficiente para rejeitar H0, não como
prova de igualdade.

## Validação

- Ground Truth facial: duas anotações humanas, consenso, IoU, precisão,
  revocação, F1 e bootstrap.
- Baseline humana: kappa ponderado entre R1/R2 e Spearman com cada margem CLIP.
- FairFace: pseudo-oráculo aplicado aos recortes do Ground Truth, com
  concordância, kappa, macro-F1 e matriz de confusão.
- Simpson: medida convergente secundária sobre categorias faciais; fica ausente
  quando há menos de dois rostos.

FairFace é uma referência secundária, não uma verdade demográfica.

## Dashboard

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

O Streamlit carrega os artefatos validados de `data/schema2/`, filtra as duas
tabelas pelos mesmos `imagem_id`, separa KPIs de imagens e rostos e mostra as
duas margens com Mann–Whitney e Wilcoxon sem replicar CLIP por face.

## Estrutura

```text
src/biasauditfw/                               # pacote testável
Framework_Auditoria_Viés_IA_Generativa.ipynb  # orquestrador Colab/T4
tests/                                         # testes e validador de outputs
data/original_64_manifest.csv                  # manifesto do corpus original
data/schema2/                                  # outputs validados do schema 2.0
app.py                                         # dashboard Streamlit
Texto_Latex/                                   # texto do TCC
openspec/                                      # especificação e tarefas
notebooks/bias_audit_pipeline.ipynb            # referência histórica
```

## Pesquisa anterior

O corpus e a baseline humana derivam de *Exploring Bias in AI-Generated
Imagery: A Design-Research Case Study in University Visual Communication*
(INTED 2026): <https://doi.org/10.21125/inted.2026.2311>.
