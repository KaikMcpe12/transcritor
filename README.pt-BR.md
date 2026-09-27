# Transcritor (WhisperX + Docker)

[English](README.md) · [Português](README.pt-BR.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Transcrição de áudio para texto com **timestamps por palavra** e **diarização opcional**
(identificação de falantes), rodando **100% offline** — via Docker ou localmente com pip.
**Não precisa de GPU**, não tem limite de duração e funciona igual em qualquer SO (Linux, macOS,
Windows) que tenha Docker.

Gera, para cada áudio:

- `transcricao_<nome>.txt` — leitura humana, em blocos por falante (quando há diarização)
- `legenda_<nome>.srt` — para Aegisub, Subtitle Edit, DaVinci Resolve, CapCut, Premiere
- `legenda_<nome>.vtt` — para HTML5/web
- `transcricao_<nome>.json` — segmentos + timestamps por palavra (uso programático)

## Funcionalidades

- 🎙️ **Transcrição + timestamps por palavra** com WhisperX
- 👥 **Diarização opcional** — ligada por padrão; use `--no-diarize` para pular
  e **rodar sem nenhum token do HuggingFace**. Guie com `--min-speakers`/`--max-speakers`
  e renomeie os rótulos com `--speakers "Alice,Bob"`
- 📁 **Modo lote** — aponte para uma pasta e ele transcreve todos os áudios, carregando os
  modelos uma única vez; `--skip-existing` retoma um lote interrompido
- 🧩 **Vários formatos de saída** — `txt`, `srt`, `vtt`, `json`, `md` (escolha com `--formats`)
- 🐳 **Docker-first** (imagem pronta no GHCR), mas também **instalável localmente** (`pip install .`)
- 💻 **Só CPU** — imagem enxuta (~2.5 GB), sem peso de CUDA
- 🌍 **Qualquer idioma** que o Whisper suporta — `--language auto` detecta (padrão: `pt`)

## Requisitos

- Docker + Docker Compose (Docker Desktop no Windows/Mac, ou `docker` +
  `docker-compose-plugin` no Linux) — **ou** Python 3.9+ e `ffmpeg` para uso local
- ~4 GB livres pra imagem + ~2 GB pros modelos (cache persistente)
- Token gratuito do HuggingFace **só se quiser a diarização**

## Início rápido (Docker)

### 1. Clonar

```bash
git clone https://github.com/KaikMcpe12/transcritor.git
cd transcritor
```

### 2. (Opcional) Token do HuggingFace — só para diarização

Pule este passo se for usar `--no-diarize`.

1. Criar conta grátis em https://huggingface.co
2. **Aceitar os termos** dos dois modelos (só clicar em "Agree and access repository"):
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-3.1
3. Gerar token do tipo **Read** em https://huggingface.co/settings/tokens
4. Copiar o template e colar o token:

```bash
cp .env.example .env
# edite .env e troque hf_cole_seu_token_aqui pelo seu token
```

### 3. Buildar a imagem (uma vez)

```bash
docker compose build
```

Demora ~5-10 min na primeira vez (baixa torch CPU + whisperx). A imagem final fica em torno de
2.5 GB (contra ~6 GB se tivesse CUDA).

### 4. Transcrever

Copie o áudio para `./audio/` e rode:

```bash
# um arquivo (com diarização, precisa do token)
docker compose run --rm transcritor /audio/meu-arquivo.mp3

# sem diarização — não precisa de token
docker compose run --rm transcritor --no-diarize /audio/meu-arquivo.mp3

# a pasta inteira de uma vez
docker compose run --rm transcritor /audio
```

Os arquivos gerados aparecem em `./output/`.

## Imagem pronta (sem build)

Não quer esperar o `docker compose build`? Baixe a imagem publicada no GitHub Container
Registry e rode direto (as flags vão depois do nome da imagem, antes do caminho do áudio):

```bash
docker run --rm \
  -v "$PWD/audio:/audio:ro" -v "$PWD/output:/output" -v "$PWD/models:/models" \
  ghcr.io/kaikmcpe12/transcritor:latest --no-diarize /audio/meu-arquivo.mp3
```

Para diarização, acrescente `-e HF_TOKEN=seu_token`. As imagens são publicadas automaticamente
pelo GitHub Actions a cada release/tag (veja `.github/workflows/docker-publish.yml`).

## Opções

Todas as flags têm variável de ambiente equivalente, então o comando clássico
`docker compose run` continua funcionando sem mudanças.

| Flag | Variável | Padrão | Descrição |
|------|----------|--------|-----------|
| `input` (posicional) | — | — | Arquivo de áudio **ou** pasta |
| `--model` | `WHISPER_MODEL` | `small` | `tiny` \| `base` \| `small` \| `medium` \| `large-v3` |
| `--language` | `LANGUAGE` | `pt` | Código ISO 639-1, ou `auto` para detectar |
| `--output` | `OUTPUT_DIR` | `.` | Diretório de saída |
| `--formats` | `FORMATS` | `txt,srt,vtt,json` | Subconjunto de `txt,srt,vtt,json,md`, por vírgula |
| `--no-diarize` | — | desligado | Não identificar falantes (dispensa o token) |
| `--speakers` | — | — | Renomeia `SPEAKER_00,01,…` na ordem, ex.: `"Alice,Bob"` |
| `--min-speakers` | — | — | Número mínimo de falantes (melhora a diarização) |
| `--max-speakers` | — | — | Número máximo de falantes |
| `--skip-existing` | — | desligado | Em modo pasta, pula áudios cujas saídas já existem |
| `--batch-size` | `BATCH_SIZE` | `8` | Reduza se faltar RAM |
| `--hf-token` | `HF_TOKEN` | — | Token do HuggingFace para diarização |

Exemplos:

```bash
docker compose run --rm transcritor --model medium --formats srt,json,md /audio/x.mp3
docker compose run --rm transcritor --speakers "Alice,Bob" --max-speakers 2 /audio/x.mp3
docker compose run --rm transcritor --language auto --skip-existing /audio
```

### Modelos

| Modelo | Tamanho | Qualidade PT-BR | Tempo em CPU (31 min de áudio) |
|--------|---------|-----------------|-------------------------------|
| small     | ~500 MB | Decente         | 15-25 min |
| medium    | ~1.5 GB | Boa             | 40-60 min |
| large-v3  | ~3 GB   | Excelente       | 90-150 min |

## Uso local (sem Docker)

Você precisa do `ffmpeg` no PATH e Python 3.9+.

```bash
# 1. PyTorch só de CPU (evita ~4 GB de libs CUDA)
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cpu

# 2. a ferramenta (instala whisperx + o comando `transcritor`)
pip install .

# rodar
transcritor meu-arquivo.mp3 --no-diarize --output ./output
transcritor ./audio --model medium
```

Como alternativa, `pip install -r requirements.txt` e rode `python transcrever.py ...`
diretamente. O Docker segue sendo o caminho mais reproduzível; a instalação local é
best-effort e depende dos wheels da sua plataforma.

## Estrutura

```
transcritor/
├── transcrever.py          # pipeline (transcrição → align → diarização opcional)
├── Dockerfile              # imagem CPU-only (torch sem CUDA)
├── docker-compose.yml      # volumes + env
├── requirements.txt        # dependências para uso local (sem Docker)
├── pyproject.toml          # instalação local + comando `transcritor`
├── .env.example            # template das variáveis
├── audio/                  # DROP: áudios de entrada (montado read-only)
├── output/                 # arquivos gerados
└── models/                 # cache dos modelos HF (persiste entre runs)
```

Os três diretórios (`audio/`, `output/`, `models/`) são montados como volumes, então áudios não
são copiados pra dentro da imagem, os modelos baixam uma vez só e os resultados aparecem direto
na máquina.

## Detalhe técnico (por que este Dockerfile)

O `pip install whisperx` puro tenta baixar `torch~=2.8.0` do PyPI padrão, que inclui **~4 GB de
bibliotecas NVIDIA/CUDA** (cudnn, cublas, nccl, cusparselt, etc.) inúteis pra CPU. Numa máquina
sem espaço, isso trava com `Errno 28`.

A solução: instalar `torch==2.8.0` do **índice CPU do PyTorch** primeiro (~200 MB, sem CUDA) e
depois `whisperx` com `--extra-index-url` apontando pro mesmo índice. Como o `2.8.0+cpu` já
satisfaz `torch~=2.8.0`, o pip não reinstala. Resultado: imagem ~2.5 GB em vez de ~6 GB.

## Renomear falantes

Com diarização, os falantes vêm como `SPEAKER_00`, `SPEAKER_01`… O jeito mais fácil de nomeá-los
é já na execução, quando você sabe quem fala primeiro:

```bash
docker compose run --rm transcritor --speakers "Kaik,Professor" /audio/x.mp3
```

Ou trocar depois, nos arquivos gerados:

```bash
cd output
sed -i 's/SPEAKER_00/Kaik/g; s/SPEAKER_01/Professor/g' transcricao_x.txt legenda_x.srt legenda_x.vtt
```

## Troubleshooting

- **Aviso de `HF_TOKEN` não definido** — esperado se você não passou token; ele só roda sem
  diarização. Defina `HF_TOKEN` ou passe `--no-diarize` para silenciar.
- **`403 Client Error` ao baixar diarização** — você não aceitou os termos dos modelos pyannote
  no HuggingFace. Voltar ao passo 2 do Início rápido.
- **`Killed` no meio da execução** — RAM insuficiente. Reduzir `--batch-size 4` ou usar modelo
  menor.
- **Muito lento** — normal em CPU. `small` é o melhor equilíbrio pra máquinas modestas.

## Licença

[MIT](LICENSE) © 2026 KaikMcpe12
