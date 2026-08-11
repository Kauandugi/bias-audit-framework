# Google Colab CLI no Windows

Este projeto usa o Google Colab CLI oficial dentro do Ubuntu 24.04 no WSL. O
CLI não oferece suporte nativo ao Windows, por isso os comandos PowerShell são
encaminhados para a distribuição Linux.

## Instalação

No PowerShell com permissão administrativa:

```powershell
wsl --install -d Ubuntu-24.04
```

Crie um usuário Linux dedicado e defina-o como padrão. Este repositório usa
`kauandugi`; o nome pode ser trocado em outra máquina:

```powershell
wsl -d Ubuntu-24.04 -u root -- useradd --create-home --shell /bin/bash kauandugi
wsl -d Ubuntu-24.04 -u root -- bash -lc "printf '[boot]\nsystemd=true\n[user]\ndefault=kauandugi\n' > /etc/wsl.conf"
wsl --shutdown
```

Instale primeiro os pacotes do sistema pelo PowerShell:

```powershell
wsl -d Ubuntu-24.04 -u root -- apt-get update
wsl -d Ubuntu-24.04 -u root -- apt-get install -y curl git python3 python3-venv ca-certificates
```

Depois instale o CLI no perfil do usuário Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv tool install google-colab-cli
```

A versão `0.6.0` publicada no PyPI pode resolver uma distribuição incompatível
de `jupyter-kernel-client`. O projeto oficial do Google usa a dependência a
partir do próprio repositório. Execute o reparo versionado pelo wrapper:

```powershell
.\tools\colab.ps1 repair-cli
.\tools\colab.ps1 doctor
```

O `doctor` deve terminar com `Kernel client: OK`. A revisão usada pelo wrapper
é fixa para que duas instalações recuperem a mesma implementação.

As credenciais ficam em `~/.config/colab-cli/` dentro do WSL. Esse diretório
nunca deve ser copiado para o repositório.

## Primeiro acesso

Execute no PowerShell:

```powershell
.\tools\colab.ps1 doctor
.\tools\colab.ps1 auth
```

O comando `auth` imprime uma URL. Entre com a mesma conta usada no Colab, copie
o código OAuth completo exibido na página final e cole no terminal. Um código
de seis dígitos da autenticação em duas etapas não é o código OAuth.

## Teste T4

```powershell
.\tools\colab.ps1 gpu-probe
```

O teste deve informar `cuda_available: true` e uma GPU T4. A disponibilidade
depende da cota e do plano da conta.

## Execução do BiasAuditFW

```powershell
.\tools\colab.ps1 start
.\tools\colab.ps1 mount-drive
.\tools\colab.ps1 smoke
```

O `mount-drive` exige confirmação humana. O comando `smoke` executa o notebook,
exporta logs para `execucao/colab/` e encerra a VM mesmo quando ocorre erro.
Depois de validar o smoke test, repita `start`, `mount-drive` e use `full`.

Para inspecionar ou encerrar manualmente:

```powershell
.\tools\colab.ps1 status
.\tools\colab.ps1 stop
```

O notebook automatizado aceita `BIASAUDIT_REPOSITORY_REF`,
`BIASAUDIT_DATASET_ROOT`, `BIASAUDIT_OUTPUT_ROOT` e as flags `BIASAUDIT_RUN_*`.
O wrapper injeta a branch e o modo de execução sem editar células manualmente.

Sessões ativas consomem recursos do Colab. Não deixe uma sessão aberta após a
execução.

## Limites do MCP e das visualizações

O `colab-mcp` ainda não é configurado no Codex porque depende de atualização
dinâmica da lista de ferramentas MCP. O suporte será reavaliado quando essa
compatibilidade estiver disponível.

Visualization Mode e Fullscreen Output Sharing são recursos da interface do
Colab. Eles podem apresentar gráficos já validados, mas não substituem o código
estatístico versionado nem o Streamlit.
