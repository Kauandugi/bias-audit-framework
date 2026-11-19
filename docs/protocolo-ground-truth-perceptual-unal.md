# Protocolo legado de validação perceptual - UNAL

Este arquivo é mantido apenas como registro histórico. A coleta na UNAL não
será realizada. O protocolo vigente está em
`docs/protocolo-validacao-perceptual-unesp.md`.

## Objetivo e limite

O formulário compara avaliações humanas com indicadores automáticos de
diversidade racial percebida e diversidade de apresentação de gênero. As notas
não definem identidade, nacionalidade, ancestralidade, sexo biológico ou origem
latino-americana das pessoas representadas.

A coleta só deve começar depois de o coordenador confirmar consentimento,
anonimização e eventual exigência ética institucional. O formulário não deve
coletar nome, e-mail, matrícula ou outro identificador pessoal.

## Amostra

- 24 imagens: 12 do corpus original e 12 do corpus parceiro.
- 16 imagens de amostragem estratificada aleatória e 8 casos de extremos ou
  discordância entre métodos.
- 12 prompts neutros e 12 inclusivos, ocultos dos participantes.
- Mesmas 24 imagens para todos, distribuídas em duas ordens contrabalanceadas.
- Entre 4 e 7 participantes; menos de 4 respostas válidas caracteriza apenas
  um piloto.

## Texto inicial em espanhol

> Este formulario forma parte de un estudio sobre representaciones visuales en
> imágenes generadas por inteligencia artificial. No se solicita identificar la
> raza, nacionalidad, ascendencia, sexo biológico ni identidad de ninguna
> persona. Evalúe solamente la diversidad visual percibida dentro del grupo que
> aparece en cada imagen. Su participación es voluntaria y las respuestas se
> analizarán de forma agregada, sin recopilar nombre ni correo electrónico.

## Perguntas por imagem

**Diversidad racial percibida**

> Considerando solamente la variación visual de tonos de piel y apariencias
> racializadas dentro del grupo, ¿cuánta diversidad percibe en esta imagen?

Escala: `1 Muy baja`, `2 Baja`, `3 Intermedia`, `4 Alta`, `5 Muy alta`,
`No es posible evaluar`.

**Diversidad de presentación de género percibida**

> Considerando solamente la variación visual de presentaciones masculinas,
> femeninas y andróginas dentro del grupo, ¿cuánta diversidad percibe en esta
> imagen?

Escala: `1 Muy baja`, `2 Baja`, `3 Intermedia`, `4 Alta`, `5 Muy alta`,
`No es posible evaluar`.

## Tratamento dos dados

As respostas individuais permanecem em armazenamento privado. O script
`scripts/aggregate_human_validation.py` exporta apenas mediana, quartis, número
de respostas e alfa de Krippendorff ordinal por dimensão. Somente esses
artefatos agregados podem entrar no repositório.
