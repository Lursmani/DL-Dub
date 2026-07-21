# syntax=docker/dockerfile:1
# Two build targets (pick with --target; default = full):
#
#   core  (~1.5 GB) — GUI + every stage except separate/transcribe, for the
#                    hybrid workflow where Analyze runs in Colab.
#   full  (~5 GB)  — adds Demucs + WhisperX. CPU torch by default; for NVIDIA
#                    hosts build with --build-arg TORCH_CHANNEL=cu126 and run
#                    with --gpus all (needs the NVIDIA Container Toolkit).
#
#   docker build -t dub-full .
#   docker build --target core -t dub-full:core .
#
# The container serves the Gradio GUI on port 7861 by default (lite uses
# 7860, so both can run at once); any CLI command works too:
#   docker run ... dub-full python dub.py estimate <video>

FROM python:3.12-slim AS base

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    # Bind Gradio to all interfaces so the published port is reachable.
    GRADIO_SERVER_NAME=0.0.0.0 \
    # Model downloads (Whisper, pyannote, Demucs) land here — mount a volume
    # at /cache so the ~4 GB of weights survive container restarts.
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch

RUN mkdir -p /cache
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7861
CMD ["python", "-m", "gui"]

# ---------- core: no local ML (Analyze runs in Colab) ----------
FROM base AS core
COPY . .

# ---------- full: + local ML stages ----------
FROM base AS ml-deps
# cpu (default) or a CUDA channel like cu126 — see https://pytorch.org
ARG TORCH_CHANNEL=cpu
COPY requirements-ml.txt ./
# One resolve with the torch channel as an EXTRA index: whisperx pins a torch
# version, and for the same version pip prefers the channel's local build
# (e.g. 2.8.0+cpu) over PyPI's CUDA-bundled default. Pre-installing torch
# separately does NOT work — pip re-resolves it from PyPI (13 GB image).
RUN pip install --no-cache-dir -r requirements-ml.txt \
        --extra-index-url https://download.pytorch.org/whl/${TORCH_CHANNEL}

FROM ml-deps AS full
COPY . .
