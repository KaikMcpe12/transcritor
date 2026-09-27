# Transcritor (WhisperX + Docker)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Offline audio-to-text transcription with **word-level timestamps** and **optional speaker
diarization**, running **100% offline** — either inside Docker or locally via pip. **No GPU
required**, no duration limit, works the same on any OS (Linux, macOS, Windows) with Docker.

For each audio file it can generate:

- `transcricao_<name>.txt` — human-readable transcript, grouped by speaker (when diarizing)
- `legenda_<name>.srt` — subtitles for Aegisub, Subtitle Edit, DaVinci Resolve, CapCut, Premiere
- `legenda_<name>.vtt` — subtitles for HTML5 / web players
- `transcricao_<name>.json` — segments + per-word timestamps for programmatic use

## Features

- 🎙️ **Transcription + word-level timestamps** via WhisperX
- 👥 **Optional speaker diarization** — enabled by default; use `--no-diarize` to skip it
  and **run without any HuggingFace token**. Guide it with `--min-speakers`/`--max-speakers`
  and auto-rename labels with `--speakers "Alice,Bob"`
- 📁 **Batch mode** — pass a folder and it transcribes every audio file, loading the models
  once; `--skip-existing` resumes an interrupted run
- 🧩 **Multiple output formats** — `txt`, `srt`, `vtt`, `json`, `md` (pick with `--formats`)
- 🐳 **Docker-first** (prebuilt image on GHCR) but also **installable locally** (`pip install .`)
- 💻 **CPU-only** — a slim ~2.5 GB image with no CUDA bloat
- 🌍 **Any language** Whisper supports — `--language auto` detects it (defaults to `pt`)

## Requirements

- Docker + Docker Compose (Docker Desktop on Windows/Mac, or `docker` + `docker-compose-plugin`
  on Linux) — **or** Python 3.9+ and `ffmpeg` for local use
- ~4 GB free for the image + ~2 GB for the models (persistent cache)
- A free HuggingFace token **only if you want speaker diarization**

## Quick start (Docker)

### 1. Clone

```bash
git clone https://github.com/KaikMcpe12/transcritor.git
cd transcritor
```

### 2. (Optional) HuggingFace token — only for diarization

Skip this if you'll use `--no-diarize`.

1. Create a free account at https://huggingface.co
2. **Accept the terms** of both models (just click "Agree and access repository"):
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-3.1
3. Generate a **Read** token at https://huggingface.co/settings/tokens
4. Copy the template and paste your token:

```bash
cp .env.example .env
# edit .env and replace hf_cole_seu_token_aqui with your token
```

### 3. Build the image (once)

```bash
docker compose build
```

Takes ~5-10 min the first time (downloads CPU torch + whisperx). Final image is ~2.5 GB
(vs. ~6 GB with CUDA).

### 4. Transcribe

Drop your audio into `./audio/` and run:

```bash
# single file (with diarization, needs the token)
docker compose run --rm transcritor /audio/my-file.mp3

# without diarization — no token needed
docker compose run --rm transcritor --no-diarize /audio/my-file.mp3

# whole folder at once
docker compose run --rm transcritor /audio
```

Generated files land in `./output/`.

## Prebuilt image (no build)

Don't want to wait for `docker compose build`? Pull the image published to the GitHub
Container Registry and run it directly (flags go after the image name, before the audio path):

```bash
docker run --rm \
  -v "$PWD/audio:/audio:ro" -v "$PWD/output:/output" -v "$PWD/models:/models" \
  ghcr.io/kaikmcpe12/transcritor:latest --no-diarize /audio/my-file.mp3
```

For diarization, add `-e HF_TOKEN=your_token`. Images are published automatically by GitHub
Actions on each release/tag (see `.github/workflows/docker-publish.yml`).

## Options

All flags have environment-variable fallbacks, so the classic `docker compose run` invocation
keeps working unchanged.

| Flag | Env var | Default | Description |
|------|---------|---------|-------------|
| `input` (positional) | — | — | Audio file **or** folder |
| `--model` | `WHISPER_MODEL` | `small` | `tiny` \| `base` \| `small` \| `medium` \| `large-v3` |
| `--language` | `LANGUAGE` | `pt` | ISO 639-1 code, or `auto` to detect |
| `--output` | `OUTPUT_DIR` | `.` | Output directory |
| `--formats` | `FORMATS` | `txt,srt,vtt,json` | Comma-separated subset of `txt,srt,vtt,json,md` |
| `--no-diarize` | — | off | Skip speaker ID (no HuggingFace token required) |
| `--speakers` | — | — | Rename `SPEAKER_00,01,…` in order, e.g. `"Alice,Bob"` |
| `--min-speakers` | — | — | Minimum number of speakers (improves diarization) |
| `--max-speakers` | — | — | Maximum number of speakers |
| `--skip-existing` | — | off | In folder mode, skip files whose outputs already exist |
| `--batch-size` | `BATCH_SIZE` | `8` | Lower it if you run out of RAM |
| `--hf-token` | `HF_TOKEN` | — | HuggingFace token for diarization |

Examples:

```bash
docker compose run --rm transcritor --model medium --formats srt,json,md /audio/x.mp3
docker compose run --rm transcritor --speakers "Alice,Bob" --max-speakers 2 /audio/x.mp3
docker compose run --rm transcritor --language auto --skip-existing /audio
```

### Models

| Model | Size | PT-BR quality | Time on CPU (31 min of audio) |
|-------|------|---------------|-------------------------------|
| small     | ~500 MB | Decent     | 15-25 min |
| medium    | ~1.5 GB | Good       | 40-60 min |
| large-v3  | ~3 GB   | Excellent  | 90-150 min |

## Local use (without Docker)

You need `ffmpeg` on your PATH and Python 3.9+.

```bash
# 1. CPU-only PyTorch (avoids ~4 GB of CUDA libs)
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cpu

# 2. the tool (installs whisperx + the `transcritor` command)
pip install .

# run it
transcritor my-file.mp3 --no-diarize --output ./output
transcritor ./audio --model medium
```

Alternatively, `pip install -r requirements.txt` and run `python transcrever.py ...` directly.
Docker remains the most reproducible path; local install is best-effort and depends on your
platform's wheels.

## Project layout

```
transcritor/
├── transcrever.py          # pipeline (transcription → align → optional diarization)
├── Dockerfile              # CPU-only image (torch without CUDA)
├── docker-compose.yml      # volumes + env
├── requirements.txt        # local (non-Docker) deps
├── pyproject.toml          # local install + `transcritor` console command
├── .env.example            # env template
├── audio/                  # DROP: input audio (mounted read-only)
├── output/                 # generated files
└── models/                 # HF model cache (persists between runs)
```

The three directories (`audio/`, `output/`, `models/`) are mounted as volumes, so audio is never
copied into the image, models are downloaded once, and results appear directly on your machine.

## Why this Dockerfile

A plain `pip install whisperx` pulls `torch~=2.8.0` from the default PyPI, which includes **~4 GB
of NVIDIA/CUDA libraries** (cudnn, cublas, nccl, cusparselt, …) that are useless on CPU — enough
to blow up disk with `Errno 28` on a small machine.

The fix: install `torch==2.8.0` from **PyTorch's CPU index** first (~200 MB, no CUDA), then install
`whisperx` with `--extra-index-url` pointing at the same index. Since `2.8.0+cpu` already satisfies
`torch~=2.8.0`, pip doesn't reinstall it. Result: ~2.5 GB image instead of ~6 GB.

## Renaming speakers

With diarization, speakers come out as `SPEAKER_00`, `SPEAKER_01`… The easiest way to name them
is at run time, once you know who speaks first:

```bash
docker compose run --rm transcritor --speakers "Alice,Bob" /audio/x.mp3
```

Or rename them afterwards in the generated files:

```bash
cd output
sed -i 's/SPEAKER_00/Alice/g; s/SPEAKER_01/Bob/g' transcricao_x.txt legenda_x.srt legenda_x.vtt
```

## Troubleshooting

- **Warning about `HF_TOKEN` not set** — expected if you didn't provide a token; it just runs
  without diarization. Set `HF_TOKEN` or pass `--no-diarize` to silence it.
- **`403 Client Error` while downloading diarization** — you didn't accept the pyannote model
  terms on HuggingFace. Go back to step 2 of the Quick start.
- **`Killed` mid-run** — not enough RAM. Lower `--batch-size 4` or use a smaller model.
- **Very slow** — normal on CPU. `small` is the best balance for modest machines.

## License

[MIT](LICENSE) © 2026 KaikMcpe12
