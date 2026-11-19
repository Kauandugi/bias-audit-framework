# Protocolo da validação perceptual humana na UNESP

## Estado do instrumento

Este protocolo descreve o formulário aplicado a estudantes do último ano da
graduação em Computação da UNESP:
<https://forms.gle/APtqTNFSQm3WVHYY6>. Ele substitui o desenho anterior de
quatro formulários contrabalanceados. O projeto e o instrumento receberam
parecer final favorável do Comitê de Ética em Pesquisa da UEMS: Parecer
Consubstanciado número 8.854.629, de 5 de novembro de 2026, CAAE
02077126.7.0000.8030. Esse parecer é a referência ética adotada no trabalho
para sustentar o formulário e o uso acadêmico de seus resultados agregados. A
UNESP foi o local efetivo de recrutamento, conforme a orientação acadêmica
recebida pelo pesquisador. O texto havia sido preparado para recrutamento na
UNAL, mas, diante do prazo, o mesmo instrumento foi aplicado na UNESP sem
alteração retroativa do formulário.

A coleta ocorreu entre 9 e 15 de outubro de 2026, antes da emissão do parecer
final. Essa cronologia permanece registrada de forma transparente. A
autorização específica para o uso acadêmico agregado das respostas está
arquivada de forma privada. O parecer e a autorização não são publicados no
repositório.

A exportação contém 72 registros. Setenta participantes aceitaram o termo e
preencheram todas as grades; duas pessoas marcaram que não concordavam em
participar e não forneceram notas. O pipeline exclui esses dois registros antes
de criar códigos locais ou calcular qualquer estatística.

O formulário público possui uma única sequência fixa com 48 imagens. Cada
imagem é avaliada numa grade obrigatória de 1 a 5. Não há opção de abstenção. A
atividade informa duração estimada de cerca de 30 minutos.

## Objetivo e limites

A coleta registra como estudantes percebem representações em imagens
sintéticas. As respostas não estabelecem identidade, nacionalidade,
ancestralidade, raça ou gênero reais das pessoas representadas.

As dimensões primárias são:

1. diversidade racial percebida;
2. equilíbrio de apresentação de gênero percebido.

As dimensões exploratórias são:

3. contexto latino-americano percebido na cena;
4. diversidade corporal percebida;
5. qualidade e coerência visual da imagem.

As dimensões exploratórias serão apresentadas de forma descritiva. Elas não
entram na família principal de hipóteses das margens CLIP.

## Participantes e privacidade

- Público: estudantes do último ano da graduação em Computação da UNESP com 18
  anos ou mais.
- Participação voluntária, condicionada ao consentimento apresentado no início.
- O formulário não deve registrar nome, matrícula ou endereço de e-mail nas
  respostas.
- Respostas individuais e dados de perfil permanecem em armazenamento privado.
- Estatísticas agregadas por `imagem_id` podem integrar o repositório e o texto
  do TCC. Respostas individuais, perfis, consentimentos e códigos locais dos
  participantes permanecem privados.

As configurações administrativas do Google Forms devem ser verificadas pelo
responsável antes da liberação. O fato de a página pública informar que o e-mail
não faz parte da resposta não substitui essa conferência.

## Instrumento observado

O formulário contém 51 páginas: consentimento, perfil, instruções e 48 páginas
de imagens. Cada página de imagem apresenta uma figura e uma grade com cinco
linhas. Todas as linhas exigem uma nota inteira entre 1 e 5.

As perguntas de perfil observadas são país, cor ou raça autodeclarada segundo
categorias do IBGE, gênero autodeclarado e faixa etária. Não há pergunta sobre
instituição.
Esses campos não serão exportados para a tabela de análise do framework. Eles
podem ser descritos separadamente pelo responsável pela pesquisa, desde que o
plano ético permita e não haja risco de reidentificação.

## Condições observadas e limitações

1. O termo menciona UEMS e UNAL. A aplicação prática ocorreu na UNESP, sem
   edição posterior do texto. Essa diferença entre o instrumento e o local de
   recrutamento é registrada como limitação do procedimento.
2. O mapa privado reconciliou as 48 posições. As imagens P4 inclusivas de
   Midjourney e Whisk são arquivos PNG/JPEG válidos com nomes terminados em
   `._`; após a correção da descoberta, passam a integrar o corpus canônico de
   64 imagens e deixam de ser suplementos.
3. A ordem é igual para todos os participantes. Portanto, efeito de posição e
   fadiga não pode ser separado do efeito da imagem.
4. Como não existe a opção `Não é possível avaliar`, notas forçadas em cenas
   ambíguas devem ser reconhecidas como limitação do instrumento.

## Código privado das imagens

O mapa preenchido está em `data/private/unesp-2026/` e não é publicado. A
reconciliação usa o hash do conteúdo e a combinação única de ano, gerador,
prompt e variante presente na planilha de mapeamento e nos manifestos
congelados. O campo
`verification_method` torna essa diferença explícita.

Campos mínimos:

```text
form_code, form_position, imagem_id, dataset_id, modelo_ia,
prompt_id, pair_id, tipo_prompt
```

## Tratamento das respostas

A planilha bruta do Google Forms permanece fora do GitHub. O comando abaixo
transforma somente respostas consentidas, descarta carimbo de data, conta
Google e perfil, e produz artefatos agregados:

```powershell
py -3.12 scripts/prepare_unesp_human_ratings.py `
  --responses-xlsx CAMINHO_PRIVADO/respostas.xlsx `
  --image-map CAMINHO_PRIVADO/mapa_48_imagens.csv `
  --private-long-csv CAMINHO_PRIVADO/human_ratings_long.csv `
  --public-output-dir data/human-unesp-aggregate `
  --audit-images data/schema2/audit_images.csv data/schema2-vitor/audit_images.csv `
  --simpson-by-image data/schema2/deepface_simpson_by_image.csv `
    data/schema2-vitor/deepface_simpson_by_image.csv `
  --authorized-for-aggregate-use `
  --authorization-basis "AUTORIZACAO_INSTITUCIONAL_REGISTRADA" `
  --ethics-opinion-number "8.854.629" `
  --ethics-opinion-date "2026-11-05"
```

O importador exige 48 posições únicas, cinco notas válidas por imagem e um
`imagem_id` canônico por posição. Ele gera códigos locais `P001`, `P002` e assim
por diante. O arquivo individual continua privado.

Com a autorização registrada, os artefatos públicos são:

- `human_ratings_aggregate.csv`, com mediana, primeiro quartil, terceiro quartil
  e número de respostas por imagem e dimensão;
- `human_interrater_agreement.csv`, com alfa de Krippendorff ordinal;
- `human_paired_prompt_effects.csv`, com Wilcoxon e famílias de Holm separadas
  por corpus e papel analítico;
- `human_framework_convergence.csv`, com Spearman, bootstrap, Holm e análise de
  sensibilidade do Simpson ao número de faces;
- `human_validation_metadata.json`, sem dados de perfil;
- `human_judge_convergence.csv`, somente quando existirem saídas MLLM completas
  e aprovadas pelo gate.

Raça e gênero foram comparados às margens CLIP correspondentes por Spearman com
bootstrap. Contexto, corpo e qualidade foram analisados em família exploratória
separada. A convergência com Simpson é secundária e inclui correlação parcial
por postos controlando o número de faces detectadas. Após as exclusões, as 70
respostas válidas produziram 3.360 avaliações participante--imagem e 16.800
notas individuais nas cinco dimensões.

## Checklist de liberação

- [x] uso acadêmico dos agregados autorizado pelo orientador e pela coordenadora do Comitê de Ética;
- [x] Parecer Consubstanciado número 8.854.629, de 5 de novembro de 2026, arquivado em local privado;
- [x] parecer final favorável vinculado ao CAAE 02077126.7.0000.8030;
- [x] 48 imagens mapeadas; após a correção, todas possuem correspondência canônica;
- [x] ausência de nomes de gerador, corpus e tipo de prompt na avaliação;
- [x] planilha de respostas mantida fora do GitHub;
- [x] recusas de consentimento excluídas antes da análise;
- [x] limitações da ordem fixa e da ausência de abstenção registradas.
- [x] somente agregados por `imagem_id` liberados para o texto, dashboard e checkpoint;
- [x] respostas individuais, perfis, consentimentos e mapa privado mantidos fora do GitHub.
