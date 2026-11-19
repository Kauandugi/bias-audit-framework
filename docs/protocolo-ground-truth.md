# Protocolo de Ground Truth para detecção facial

> **Documento legado.** Este protocolo não foi executado e não descreve a
> validação vigente. A detecção foi avaliada no WIDER FACE e comparada ao MTCNN
> no corpus sintético. A coleta humana atual é perceptual e está descrita em
> `docs/protocolo-validacao-perceptual-unesp.md`.

## Objetivo

Medir se o RetinaFace localizou corretamente os rostos presentes nas 64 imagens sintéticas. O protocolo avalia detecção geométrica; não atribui identidade, nacionalidade, raça ou gênero às pessoas representadas.

## Avaliadores

Dois avaliadores anotam as imagens de forma independente, sem consultar as caixas produzidas pelo modelo. Cada anotador recebe a mesma lista ordenada de arquivos e usa apenas a classe `face`.

## Regra de anotação

- Desenhar uma caixa para cada face humana visível com elementos suficientes para reconhecê-la como face.
- Incluir faces parcialmente ocluídas quando olhos, nariz ou boca permitirem a identificação visual.
- Não anotar reflexos, fotografias em cartazes, esculturas ou padrões que apenas se pareçam com rostos.
- Marcar como ambígua uma região que não possa ser decidida com segurança.
- Ajustar a caixa ao rosto, sem incluir tronco ou cabelo em excesso.

## Ferramenta e exportação

Usar CVAT ou ferramenta equivalente e exportar cada anotação em COCO JSON. Os arquivos esperados são `ground_truth_avaliador_1.json` e `ground_truth_avaliador_2.json`.

## Consenso

As caixas dos avaliadores são pareadas por maior IoU. Pares com IoU igual ou superior a 0,5 e mesma contagem são aceitos. Diferenças de contagem, caixas sem par ou IoU inferior a 0,5 são revistas em conjunto. O consenso é exportado como `ground_truth_consenso.json` e convertido pelo notebook em `ground_truth_faces.csv`.

## Critério de acerto do detector

Uma predição do RetinaFace é verdadeiro positivo quando corresponde a uma caixa do consenso com IoU igual ou superior a 0,5. Cada caixa pode participar de apenas um par. Predições sem par são falsos positivos; caixas do consenso sem predição são falsos negativos.

## Métricas

O relatório apresenta precisão, revocação, F1, média de IoU dos pares válidos, falsos positivos e falsos negativos. As métricas são calculadas no conjunto completo e separadas por modelo gerador e tipo de prompt. Intervalos de confiança são estimados por bootstrap no nível da imagem.

## Limite da avaliação

O Ground Truth responde se a região facial foi detectada. Ele não confirma as categorias demográficas retornadas pelo DeepFace. Essas categorias serão comparadas com o FairFace como análise de concordância, não como validação de identidade real.
