FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG COSYVOICE_REF=074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    COSYVOICE_REPO=/opt/CosyVoice \
    MODEL_ID=FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    MODEL_REVISION=29e01c4e8d000f4bcd70751be16fa94bf3d85a18 \
    MODEL_DIR=/models/Fun-CosyVoice3-0.5B \
    DATA_DIR=/data \
    WHISPER_CACHE_DIR=/models/whisper \
    REQUIRE_CUDA=1 \
    COSYVOICE_FP16=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ffmpeg \
        git \
        git-lfs \
        libsndfile1 \
        libsox-dev \
        ninja-build \
        python-is-python3 \
        python3-dev \
        python3-pip \
        sox \
        unzip \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice \
    && git -C /opt/CosyVoice checkout "${COSYVOICE_REF}" \
    && git -C /opt/CosyVoice submodule update --init --recursive \
    && python -m pip install pip==25.3 setuptools==80.9.0 wheel==0.45.1 \
    && python -m pip install --no-build-isolation --no-deps openai-whisper==20231117 \
    && python -m pip install -r /opt/CosyVoice/requirements.txt

WORKDIR /workspace
COPY requirements-app.txt ./requirements-app.txt
RUN python -m pip install -r requirements-app.txt

COPY app.py ./app.py
COPY voice_tts ./voice_tts

VOLUME ["/models", "/data"]
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
    CMD curl --fail --silent --user "${GRADIO_USERNAME}:${GRADIO_PASSWORD}" http://127.0.0.1:7860/ > /dev/null || exit 1

CMD ["python", "app.py"]
