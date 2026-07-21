# syntax=docker/dockerfile:1
# API-only auto-dub image (~1.5 GB). Serves the Gradio GUI on port 7860 by
# default; any CLI command works too:
#   docker run ... dub-lite python autodub.py input/episode.mp4 --yes

FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    # Bind Gradio to all interfaces so the published port is reachable.
    GRADIO_SERVER_NAME=0.0.0.0

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["python", "-m", "gui"]
