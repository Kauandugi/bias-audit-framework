## Contexto

O corpus original contém 64 imagens, quatro modelos geradores e pares de prompts
neutros e inclusivos. O projeto também deverá receber datasets futuros cuja
organização de pastas ainda é desconhecida. A unidade semântica é a imagem; a
unidade facial é cada rosto detectado.

## Decisões

### Pacote e orquestração

O código reutilizável ficará em `src/biasauditfw`. O notebook instalará esse
pacote, montará o Drive, fará preflight, executará testes, carregará os modelos
uma vez e então rodará smoke test ou corpus completo.

### Ingestão

A descoberta será recursiva e independente da profundidade das pastas. O
pipeline validará imagens com Pillow, ignorará links simbólicos e o diretório de
saída, calculará SHA-256 e rejeitará conteúdo duplicado por padrão.

Metadados analíticos virão de um manifesto opcional com `relative_path`
obrigatório. Sem manifesto, o processamento continua, mas os testes inferenciais
são ignorados com motivo explícito.

### CLIP multidimensional

Serão usados quatro protótipos: diversidade e homogeneidade racial percebida;
diversidade e homogeneidade de apresentação de gênero. Cada protótipo agrega três
frases. Os vetores das frases são normalizados por L2, promediados e normalizados
novamente. O vetor da imagem também é normalizado por L2.

As margens são:

```text
racial_margin = cos(image, racial_diversity) - cos(image, racial_homogeneity)
gender_margin = cos(image, gender_diversity) - cos(image, gender_homogeneity)
```

Não haverá softmax, contagem por CLIP ou escore único que misture as dimensões.

### Dados e estatística

`audit_images.csv` terá uma linha por imagem e os seis valores CLIP.
`audit_faces.csv` terá uma linha por rosto e nenhuma coluna CLIP. Mann–Whitney e
Wilcoxon usarão somente linhas de imagem elegíveis. A correção de Holm cobrirá as
duas hipóteses primárias e, em família separada, as análises por modelo.

### Validação

Não haverá anotação humana de caixas no corpus sintético. RetinaFace será
avaliado no benchmark WIDER FACE por IoU, precisão, revocação, F1 e AP50. No
corpus sintético, sua discordância com MTCNN será descrita por contagem, caixas
pareadas e detecções não correspondidas, sem chamar MTCNN de Ground Truth.
FairFace operará nos recortes do RetinaFace como pseudo-oráculo condicional.

Uma avaliação humana perceptual foi coletada com estudantes voluntários do
último ano da graduação em Computação da UNESP. O projeto e o instrumento
receberam decisão final favorável do Comitê de Ética em Pesquisa da UEMS por
meio do Parecer Consubstanciado número 8.854.629, de 5 de novembro de 2026,
vinculado ao CAAE 02077126.7.0000.8030. Esse parecer é a referência ética
adotada no trabalho para sustentar o formulário e o uso acadêmico de seus
resultados agregados. A UNESP foi o local efetivo de recrutamento, conforme a
orientação acadêmica recebida pelo pesquisador. O parecer e a autorização
específica para o uso acadêmico agregado permanecem em arquivo privado. O
instrumento havia sido preparado para recrutamento na UNAL e não foi alterado
retroativamente. A coleta ocorreu entre 9 e 15 de outubro de 2026, antes da
emissão do parecer final; essa cronologia permanece registrada de forma
transparente.
O instrumento vigente usa
um único formulário, uma sequência fixa de 48 imagens e cinco notas ordinais de
1 a 5 por imagem. Diversidade racial percebida e equilíbrio de apresentação de
gênero são dimensões primárias. Contexto latino-americano, diversidade corporal
e qualidade visual são dimensões exploratórias. O formulário não oferece
abstenção e não possui contrabalanceamento; respostas forçadas, fadiga e ordem
fixa são registradas como limitações. A coleta recebeu 72 submissões, das quais
70 continham consentimento e respostas completas. Respostas individuais,
perfis, consentimentos e o mapa do formulário permanecem privados; apenas
agregados por imagem e relatórios derivados entram nas conclusões e no pacote
público. A expressão "referência humana perceptual" não representa identidade
demográfica objetiva.

### MLLM-as-a-Judge

O módulo opcional `biasauditfw.judges` permite executar provedores externos e
modelos locais de forma separada. O piloto canônico de oito imagens usa
Gemini, um modelo Qwen acessado pela Groq e `Qwen/Qwen2.5-VL-7B-Instruct`
local. Os adaptadores para `Qwen/Qwen3-VL-8B-Instruct` e
`google/gemma-3-4b-it` permanecem como experimentos diagnósticos separados.
Nenhuma execução modificará
`audit_images.csv` ou `audit_faces.csv`. Os juízes produzirão notas de 1 a 5 por
imagem e escolhas pareadas entre prompts neutros e inclusivos. Cada resposta
registrará modelo, versão, prompt, imagem, ordem, repetição e estado de parsing.
A ordem A/B será invertida para medir viés de posição.

A avaliação principal não solicitará justificativas. Explicações serão coletadas
em chamada separada apenas para oito casos de maior discordância. Nenhum juiz
será tratado isoladamente como verdade; estabilidade, acordo entre juízes e
convergência com humanos serão reportados.

Antes de uma execução local completa, um piloto semântico SHALL usar quatro
pares completos, selecionados deterministicamente com cobertura de geradores e
ao menos dois rostos detectados por imagem. Esse filtro facial serve apenas para
evitar um smoke test vazio e não constitui Ground Truth. O piloto rejeitará
abstenção excessiva, notas constantes, excesso de `not_evaluable`, respostas
pareadas constantes e baixa consistência após inversão A/B. A aprovação desse
gate é condição operacional, não evidência de acurácia. Qwen3-VL e Gemma 3
serão carregados separadamente em NF4 de 4 bits, com checkpoints distintos para
evitar concorrência de memória na T4. Modelo, revisão, quantização e limites de
geração permanecerão rastreáveis. O adaptador Qwen2.5 anterior será preservado
somente para reprodutibilidade histórica e diagnóstico. Um lote completo MUST
receber um CSV de amostra congelado e será executado separadamente para cada
corpus; habilitar a execução sem esse mapa deverá falhar explicitamente.

### Replicação do protocolo em corpus parceiro

Um segundo corpus preserva pares neutro-inclusivo e tem sua composição
registrada em manifesto antes da análise. Os corpora são analisados
separadamente antes de qualquer resumo conjunto. A primeira descoberta baseada
em sufixos encontrou 62 imagens. A inspeção de conteúdo identificou ainda as
imagens inclusivas P4 de Midjourney e Whisk, uma PNG e uma JPEG válidas cujos
nomes terminavam em `._`. Após validação pelo Pillow, o corpus foi corrigido para
64 imagens organizadas em 32 pares completos. O segundo corpus
replica o protocolo, mas não é tratado como amostra estatisticamente
independente nem como validação em larga escala.

## Riscos e limites

- CLIP é sensível à formulação das frases e tem limitações de contagem e
  composição visual.
- DeepFace e FairFace classificam aparências percebidas e podem reproduzir
  vieses; não inferem identidade, ancestralidade ou sexo biológico.
- Datasets sem manifesto não suportam comparações neutro/inclusivo.
- Modelos usados como juízes podem apresentar viés de posição, estilo,
  fornecedor e auto-preferência; por isso haverá análise por juiz e
  sensibilidade por família de gerador.
- Cotas gratuitas e identificadores de APIs podem mudar; a execução falhará se
  o modelo fixado estiver indisponível, sem substituição silenciosa.
- Resultados neurais só serão considerados finais após execução no T4 e
  validação independente dos outputs.
- O formulário humano atual usa ordem fixa e não permite abstenção; por isso,
  não é possível separar completamente percepção, posição e fadiga.
