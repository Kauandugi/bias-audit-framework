# BiasAuditFW

BiasAuditFW é uma ferramenta de pesquisa para analisar como pessoas são
representadas em imagens geradas por inteligência artificial. O projeto compara
imagens criadas com instruções neutras e inclusivas em situações do cotidiano
universitário.

O trabalho começou durante um intercâmbio na Universidad Nacional de Colombia e
foi ampliado no Trabalho de Conclusão de Curso em Ciência da Computação da UEMS.

## O que o projeto investiga

A pergunta central é simples: pedir diversidade de forma explícita muda a
representação presente nas imagens?

A resposta encontrada foi parcial. As pessoas perceberam mais contexto
latino-americano e mais diversidade corporal nas imagens inclusivas. Para
diversidade racial e equilíbrio de gênero, a análise principal não encontrou
uma diferença clara. As medidas automáticas também não acompanharam as notas
humanas na maior parte das comparações.

Para estudar essa questão, o projeto analisa dois aspectos:

1. O conteúdo geral da imagem, com o modelo CLIP.
2. A composição dos rostos detectados, com RetinaFace e DeepFace.

As medições automáticas são comparadas com avaliações humanas agregadas. Essa
comparação ajuda a verificar se a ferramenta e os participantes percebem as
imagens de forma semelhante.

As categorias produzidas pelos classificadores descrevem apenas a saída dos
modelos. Elas não determinam identidade, ancestralidade, nacionalidade, sexo
biológico ou pertencimento cultural das pessoas retratadas.

## Como funciona

O fluxo principal tem cinco etapas:

1. O sistema localiza as imagens e confere se os arquivos são válidos.
2. O manifesto identifica o gerador, a instrução usada e o par de comparação.
3. O CLIP calcula medidas semânticas no nível da imagem.
4. RetinaFace detecta os rostos e DeepFace atribui categorias aos recortes.
5. Os resultados são organizados em tabelas para análise estatística e
   visualização.

O projeto aceita imagens em pastas com qualquer organização. Um manifesto em
CSV é usado para preservar a relação entre cada imagem, sua instrução e seu par
de comparação.

## Arquivos produzidos

Cada execução no formato atual gera estes arquivos principais:

| Arquivo | Conteúdo |
| --- | --- |
| `audit_images.csv` | Uma linha por imagem, com as medidas do CLIP e os filtros de análise |
| `audit_faces.csv` | Uma linha por rosto detectado, com a caixa facial e as categorias do DeepFace |
| `ingestion_report.csv` | Resultado da leitura e validação de cada arquivo encontrado |
| `run_metadata.json` | Versões, configurações e informações da execução |
| `inference_status.json` | Indicação de quais análises estatísticas podem ser executadas |

Os resultados validados do primeiro conjunto estão em
[`data/schema2/`](data/schema2/). A replicação está em
[`data/schema2-vitor/`](data/schema2-vitor/).

## Execução local

O projeto requer Python 3.12. O gerenciador `uv` é recomendado porque instala as
versões registradas no arquivo de dependências. Para baixar o código e preparar
o ambiente:

```bash
git clone https://github.com/Kauandugi/bias-audit-framework.git
cd bias-audit-framework
uv sync --python 3.12 --extra test --extra dashboard
uv run python -m pytest -q tests -m "not gpu"
```

Para processar um conjunto de imagens:

```bash
uv run --python 3.12 --extra neural --frozen python scripts/run_corpus.py \
  CAMINHO_DAS_IMAGENS \
  CAMINHO_DO_MANIFESTO.csv \
  output/runs/minha-execucao \
  --dataset-id minha-execucao \
  --expected-images 64
```

No PowerShell, use o acento grave no fim de cada linha ou escreva o comando em
uma única linha.

## Execução no Google Colab

O notebook principal é
[`Framework_Auditoria_Viés_IA_Generativa.ipynb`](Framework_Auditoria_Viés_IA_Generativa.ipynb).

1. Abra o notebook no Google Colab.
2. Selecione uma GPU T4.
3. Informe os caminhos do conjunto, do manifesto e da pasta de saída.
4. Execute primeiro o teste reduzido com `RUN_FULL = False`.
5. Confira o resultado e altere para `RUN_FULL = True`.
6. Execute o validador final.

O fluxo automatizado para Windows e WSL está explicado em
[`docs/colab-cli-workflow.md`](docs/colab-cli-workflow.md).

## Dashboard

O dashboard apresenta as medidas por imagem e por rosto. Ele também mostra os
resultados humanos agregados quando os metadados permitem esse uso.

```bash
uv run --python 3.12 --extra dashboard streamlit run app.py
```

## Testes e validação

A suíte cobre a leitura das imagens, os manifestos, os contratos das tabelas,
os cálculos estatísticos, os resultados agregados e os notebooks. A cobertura
mínima exigida para os módulos não neurais é de 85%.

Também foram feitas verificações externas:

* WIDER FACE para avaliar a detecção do RetinaFace.
* MTCNN para comparar a contagem e a localização dos rostos.
* FairFace como segunda referência algorítmica para os mesmos recortes.
* Avaliação humana para comparar percepção e medidas automáticas.

No WIDER FACE, o RetinaFace obteve precisão de 0,9762, revocação de 0,5675, F1
de 0,7177 e AP50 global de 0,5642 no protocolo usado pelo projeto. Esses valores
avaliam o detector nesse conjunto público. Eles não medem diretamente seu
desempenho nas imagens sintéticas do estudo.

## Resultados disponíveis

Os dois conjuntos somam 128 imagens organizadas em 64 pares e 998 rostos
detectados.

Na avaliação humana, 70 participantes avaliaram 48 imagens. Isso resultou em
3.360 avaliações de imagens por participantes e 16.800 notas nas cinco
dimensões do formulário. As notas de contexto latino-americano e diversidade
corporal foram maiores nas imagens inclusivas dos dois conjuntos. As duas
dimensões principais não apresentaram diferença significativa após a correção
de Holm.

As medidas do CLIP não tiveram associação significativa com as notas humanas.
As análises com avaliadores multimodais foram mantidas como pilotos e não foram
usadas para sustentar as conclusões do trabalho.

Os resultados devem ser interpretados como evidência exploratória sobre as
imagens estudadas. Eles não permitem afirmar como todos os geradores se
comportam em qualquer contexto.

## Privacidade

O repositório contém somente resultados humanos agregados. Respostas
individuais, perfis, consentimentos, endereços de e-mail e o mapa privado do
formulário não são publicados.

Os agregados autorizados estão em
[`data/human-unesp-aggregate/`](data/human-unesp-aggregate/). Os protocolos de
preparação e validação estão na pasta [`docs/`](docs/).

## Organização do repositório

```text
src/biasauditfw/                               código principal
scripts/                                       comandos de preparação e execução
tests/                                         testes e validadores
data/schema2/                                  resultados do primeiro conjunto
data/schema2-vitor/                            resultados da replicação
data/human-unesp-aggregate/                    resultados humanos agregados
output/validation/                             validações técnicas publicadas
docs/                                          protocolos e instruções detalhadas
Framework_Auditoria_Viés_IA_Generativa.ipynb  notebook principal para o Colab
app.py                                         dashboard em Streamlit
```

## Pesquisa anterior

O estudo dá continuidade ao artigo *Exploring Bias in AI-Generated Imagery: A
Design-Research Case Study in University Visual Communication*, publicado nos
anais do INTED 2026.

DOI: <https://doi.org/10.21125/inted.2026.2311>
