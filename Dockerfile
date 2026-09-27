# Transcritor com WhisperX — CPU-only, PT-BR, diarização + timestamps
# ---------------------------------------------------------------------
# O truque central: instalar torch da CPU index ANTES do whisperx, pinado
# na versão exata que ele exige (2.8.0). Assim o resolver não puxa os ~4
# GB de libs NVIDIA (cudnn, cublas, nccl, cusparselt, etc.) que fazem o
# install estourar disco em máquinas sem GPU.

FROM python:3.11-slim

# --- deps de sistema ---------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- torch CPU (fixado nas versões que o whisperx pede) ----------------
# --index-url força o wheel CPU-only do PyTorch (sem CUDA)
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.8.0 \
        torchaudio==2.8.0 \
        torchvision==0.23.0

# --- whisperx + resto das deps -----------------------------------------
# --extra-index-url mantém o CPU index como fallback caso o resolver
# reavalie torch por algum motivo
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        whisperx==3.8.6

# --- caches de modelos (persistem via volume) --------------------------
ENV HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    HF_HUB_ENABLE_HF_TRANSFER=0

COPY transcrever.py /app/transcrever.py

# Saída vai pro CWD; docker-compose monta ./output aqui
WORKDIR /output

ENTRYPOINT ["python", "/app/transcrever.py"]
