## Motivação

O BiasAuditFW precisa analisar datasets além do corpus original sem depender da
estrutura de diretórios. Também é necessário substituir o eixo CLIP
latino-americano/não latino-americano, que confundia alinhamento visual com uma
identidade heterogênea, por duas dimensões mais delimitadas: diversidade racial
percebida e diversidade de apresentação de gênero percebida.

## Mudança proposta

- Extrair o pipeline do notebook para o pacote testável `src/biasauditfw`.
- Descobrir imagens recursivamente e obter metadados apenas de um manifesto
  opcional.
- Preservar o contrato 1:N entre imagens e rostos.
- Construir quatro protótipos CLIP por conjunto de frases, com normalização L2
  antes e depois da média dos vetores textuais.
- Produzir margens independentes para diversidade racial percebida e
  apresentação de gênero percebida.
- Executar inferência estatística somente quando o manifesto identificar pares e
  grupos válidos.
- Manter Ground Truth humano, baseline humana e FairFace como validações
  complementares.
- Atualizar notebook, dashboard, documentação, OpenSpec e texto do TCC.

## Impacto

O schema canônico passa para a versão `2.0`. Outputs antigos são preservados como
legado e não são convertidos silenciosamente. O notebook oficial continua sendo
`Framework_Auditoria_Viés_IA_Generativa.ipynb`, mas passa a orquestrar o pacote.
A execução neural completa permanece no Google Colab T4.

