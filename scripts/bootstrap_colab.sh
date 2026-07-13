#!/usr/bin/env bash
set -euo pipefail

COSYVOICE_REF="${COSYVOICE_REF:-074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc}"
COSYVOICE_REPO="${COSYVOICE_REPO:-/content/CosyVoice}"
CONDA_DIR="${CONDA_DIR:-/content/voice-tts-conda}"
APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${CONDA_DIR}/bin/python"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py310_25.5.1-1-Linux-x86_64.sh"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is missing. Select an L4 GPU runtime in Colab." >&2
  exit 1
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

apt-get -qq update
apt-get -qq install -y ffmpeg sox libsox-dev libsndfile1 git git-lfs build-essential ninja-build curl unzip
git lfs install

if [[ ! -x "${PYTHON}" ]]; then
  curl -fsSL "${MINICONDA_URL}" -o /content/miniconda-installer.sh
  bash /content/miniconda-installer.sh -b -p "${CONDA_DIR}"
  rm -f /content/miniconda-installer.sh
fi

if [[ ! -d "${COSYVOICE_REPO}/.git" ]]; then
  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "${COSYVOICE_REPO}"
fi

git -C "${COSYVOICE_REPO}" fetch --depth 1 origin "${COSYVOICE_REF}"
git -C "${COSYVOICE_REPO}" checkout --detach "${COSYVOICE_REF}"
git -C "${COSYVOICE_REPO}" submodule update --init --recursive

"${PYTHON}" -m pip install --upgrade pip
"${PYTHON}" -m pip install -r "${COSYVOICE_REPO}/requirements.txt"
"${PYTHON}" -m pip install -r "${APP_ROOT}/requirements-app.txt"

"${PYTHON}" - <<'PY'
import torch
print("Python/Torch environment ready")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the isolated Python 3.10 environment")
print("gpu:", torch.cuda.get_device_name(0))
PY

echo "COLAB_PYTHON=${PYTHON}"
