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

Ground Truth COCO será associado por caminho relativo ou `source_image_id`, nunca
apenas por nome de arquivo. RetinaFace será avaliado por IoU e pareamento
um-para-um. FairFace será executado nos recortes do Ground Truth como
pseudo-oráculo, sem ser tratado como verdade demográfica. A baseline humana será
comparada às dimensões CLIP correspondentes por kappa e Spearman. Índices de
Simpson faciais serão evidência convergente secundária.

## Riscos e limites

- CLIP é sensível à formulação das frases e tem limitações de contagem e
  composição visual.
- DeepFace e FairFace classificam aparências percebidas e podem reproduzir
  vieses; não inferem identidade, ancestralidade ou sexo biológico.
- Datasets sem manifesto não suportam comparações neutro/inclusivo.
- Resultados neurais só serão considerados finais após execução no T4 e
  validação independente dos outputs.

