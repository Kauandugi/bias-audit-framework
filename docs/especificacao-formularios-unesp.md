# Especificação do formulário aplicado na UNESP

## Versão vigente

O instrumento vigente é um único Google Forms com 48 imagens:
<https://forms.gle/APtqTNFSQm3WVHYY6>. Os antigos painéis `A1`, `B1`, `A2` e
`B2` permanecem apenas como proposta histórica e não descrevem a coleta atual.

## Estrutura confirmada na página pública

- 51 páginas no total;
- uma seção de consentimento;
- uma seção de perfil;
- uma seção de instruções;
- 48 páginas de imagens;
- uma imagem por página;
- uma grade obrigatória por imagem;
- cinco linhas por grade;
- opções inteiras de 1 a 5;
- estimativa informada de aproximadamente 30 minutos.

As cinco linhas são:

1. Diversidade racial
2. Equilíbrio de gênero
3. Contexto latino-americano
4. Diversidade corporal
5. Qualidade e coerência visual

O formulário não oferece `Não é possível avaliar`. Todas as análises devem
registrar essa escolha de desenho como uma possível fonte de resposta forçada.

## Escopo analítico

`Diversidade racial` e `Equilíbrio de gênero` são as dimensões primárias porque
possuem indicadores automáticos correspondentes no CLIP. As outras três
dimensões são exploratórias. `Contexto latino-americano` avalia ambiente,
objetos e composição da cena, não identidade ou nacionalidade de pessoas.

O formulário não mede diretamente acurácia do RetinaFace, DeepFace ou FairFace.
Ele fornece uma referência perceptual para comparação com escores agregados no
nível da imagem.

## Perfil e minimização de dados

A página pública apresenta perguntas sobre país, cor ou raça autodeclarada,
gênero autodeclarado e faixa etária. Não existe pergunta sobre instituição. O pipeline de análise
ignora essas colunas e não as transfere aos artefatos públicos. Caso o estudo
precise descrever a amostra, essa etapa será feita separadamente, de forma
agregada e conforme a autorização ética.

## Rastreabilidade obrigatória

O mapa privado preenchido relaciona cada posição a
um `imagem_id` existente em `data/schema2/audit_images.csv` ou no manifesto
corrigido do corpus parceiro. As duas imagens P4 inclusivas anteriormente
omitidas são arquivos válidos com nomes terminados em `._`; após validação de
conteúdo, as 48 posições possuem correspondência canônica.

A sequência pública não revela corpus, gerador, prompt ou condição. O mapa
privado conserva essas informações para a análise posterior.

## Limitações do desenho atual

- Um único link simplifica a aplicação, mas todos recebem a mesma ordem.
- Não há contrabalanceamento para estimar viés de posição.
- Quarenta e oito páginas podem produzir fadiga.
- A ausência de abstenção exige uma nota mesmo em cenas ambíguas.
- A referência é perceptual e não representa Ground Truth demográfico.
- Os outputs automáticos do corpus parceiro precisam ser regenerados para 64
  imagens antes de qualquer comparação com as respostas.

## Situação ética e aplicação

O projeto e o instrumento foram aprovados pelo Comitê de Ética em Pesquisa da
UEMS por meio do Parecer Consubstanciado número 8.854.629, de 5 de novembro de
2026, vinculado ao CAAE 02077126.7.0000.8030. Esse parecer é a referência ética
adotada no trabalho para sustentar o formulário e o uso acadêmico de seus
resultados agregados. A UNESP foi o local efetivo de recrutamento, conforme a
orientação acadêmica recebida pelo pesquisador. Os documentos institucionais
completos e a autorização específica para o uso acadêmico agregado permanecem
em armazenamento privado. A coleta ocorreu entre 9 e 15 de outubro de 2026,
antes da emissão do parecer final; essa cronologia permanece registrada de
forma transparente.

O formulário havia sido preparado para recrutamento na UNAL. Por razão de
prazo e viabilidade, foi aplicado a estudantes da UNESP sem alteração
retroativa do instrumento. Foram recebidos 72 registros: 70 participantes
aceitaram o termo e preencheram as 48 grades; duas pessoas recusaram e não
forneceram notas. As recusas foram excluídas antes da análise. Somente
resultados agregados são publicados; respostas individuais, perfis,
consentimentos e o mapa privado das imagens permanecem fora do repositório.
