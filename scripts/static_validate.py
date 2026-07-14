from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def validate_python() -> None:
    paths = [
        ROOT / "app.py",
        *sorted((ROOT / "voice_tts").glob("*.py")),
        *sorted((ROOT / "scripts").glob("*.py")),
        *sorted((ROOT / "tests").glob("*.py")),
    ]
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        print("Python syntax OK:", path.relative_to(ROOT))


def validate_notebooks() -> None:
    notebook_dir = ROOT / "output" / "jupyter-notebook"
    notebook_paths = sorted(notebook_dir.glob("*.ipynb"))
    if [path.name for path in notebook_paths] != ["voice_tts_colab_gpu.ipynb"]:
        raise ValueError("There must be exactly one canonical Colab notebook")

    for path in notebook_paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
            raise ValueError(f"Invalid notebook structure: {path}")
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            if cell.get("execution_count") is not None or cell.get("outputs"):
                raise ValueError(f"Notebook outputs must be empty: {path.name}:cell-{index}")
            source = "".join(cell.get("source", []))
            if source.strip():
                ast.parse(source, filename=f"{path.name}:cell-{index}")
        print("Notebook structure/syntax OK:", path.relative_to(ROOT))

    canonical = json.loads((notebook_dir / "voice_tts_colab_gpu.ipynb").read_text(encoding="utf-8"))
    if canonical.get("metadata", {}).get("accelerator") != "GPU":
        raise ValueError("Canonical Colab notebook must request a GPU accelerator")
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in canonical["cells"]
        if cell.get("cell_type") == "code"
    )
    forbidden_docker_commands = [r"\[\s*['\"]docker['\"]", r"!\s*docker\b", r"\bdocker\s+compose\b"]
    if any(re.search(pattern, code, flags=re.IGNORECASE) for pattern in forbidden_docker_commands):
        raise ValueError("Hosted Colab notebook must not invoke Docker Engine")
    if '[*udocker, "pull"' in code:
        raise ValueError("Hosted Colab must use the resumable OCI downloader, not udocker pull")
    for required in [
        'EXECUTION_MODE = "udocker"',
        'ENABLE_VOXCPM = True',
        'VOXCPM_VERSION = "2.0.3"',
        'VOXCPM_MODEL_ID = "openbmb/VoxCPM2"',
        'voxcpm_worker.py',
        '"DEFAULT_TTS_ENGINE": "voxcpm2"',
        'REPO_URL = "https://github.com/nahmax/voice-tts.git"',
        'UDOCKER_VERSION = "1.3.17"',
        'f"{image_repository}:sha-{commit[:12]}"',
        '"--nvidia"',
        '"--allow-root"',
        'EXECUTION_MODE == "udocker"',
        'EXECUTION_MODE == "native"',
        "print('Model:', type(get_model()).__name__)",
        '"GRADIO_SHARE": "0"',
        'trycloudflare',
        'voice_tts_cloudflared.pid',
        'STOP_APP = False',
        'WARNING: Google Drive mount failed',
        'STORAGE_ROOT = Path("/content/voice-tts-data")',
        'pull_oci_resumable.py',
        '[*udocker, "load", "-i"',
        'OCI_ARCHIVE.unlink(missing_ok=True)',
        'PROGRESS_HEARTBEAT_SECONDS = 15',
        'GRADIO_START_TIMEOUT_SECONDS = 600',
        'class StageProgress',
        'после этого останется этапов',
        'UDOCKER_READY_MARKER.write_text',
        'NATIVE_READY_MARKER.write_text',
    ]:
        if required not in code:
            raise ValueError(f"Canonical notebook is missing: {required}")
    print("Colab OCI/GPU workflow present: voice_tts_colab_gpu.ipynb")

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    if "python -m pip uninstall -y deepspeed" not in dockerfile:
        raise ValueError("Docker image must remove the training-only DeepSpeed package")

    oci_downloader = (ROOT / "scripts" / "pull_oci_resumable.py").read_text(encoding="utf-8")
    for required in [
        "class DownloadProgress",
        "OCI download plan:",
        "remaining {format_bytes(remaining)}",
        "ETA {eta}",
    ]:
        if required not in oci_downloader:
            raise ValueError(f"OCI downloader is missing progress reporting: {required}")

    app_source = (ROOT / "app.py").read_text(encoding="utf-8")
    if "allowed_paths=[str(runs_dir(DATA_DIR))]" not in app_source:
        raise ValueError("Gradio must be allowed to serve generated WAV files")
    for required in [
        "available_tts_engines()",
        "engine=selected_engine",
        '"tts_engine": selected_engine',
        "_create_browser_preview",
        'label="Исходный WAV 48 кГц"',
        '"preview_audio": str(preview_audio)',
    ]:
        if required not in app_source:
            raise ValueError(f"Gradio app is missing multi-engine integration: {required}")

    worker_source = (ROOT / "scripts" / "voxcpm_worker.py").read_text(encoding="utf-8")
    for required in ["VoxCPM.from_pretrained", "prompt_wav_path", "reference_wav_path", "--preload"]:
        if required not in worker_source:
            raise ValueError(f"VoxCPM2 worker is missing: {required}")


def validate_text_artifacts() -> None:
    shell_path = ROOT / "scripts" / "bootstrap_colab.sh"
    shell_bytes = shell_path.read_bytes()
    if not shell_bytes.startswith(b"#!/usr/bin/env bash\n") or b"\r\n" in shell_bytes:
        raise ValueError("bootstrap_colab.sh must use a bash shebang and LF line endings")
    print("Shell artifact structure OK:", shell_path.relative_to(ROOT))

    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    for required in ["driver: nvidia", "capabilities: [gpu]", "count: 1"]:
        if required not in compose:
            raise ValueError(f"compose.yaml is missing: {required}")
    print("Compose GPU declaration present: compose.yaml")

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    for required in [
        "FROM nvidia/cuda:",
        "REQUIRE_CUDA=1",
        "COSYVOICE_FP16=1",
        "pip==25.3 setuptools==80.9.0 wheel==0.45.1",
        "--no-build-isolation --no-deps openai-whisper==20231117",
    ]:
        if required not in dockerfile:
            raise ValueError(f"Dockerfile is missing: {required}")
    if "GRADIO_PASSWORD=" in dockerfile:
        raise ValueError("Dockerfile must not bake a Gradio password into image ENV")
    print("Docker CUDA runtime declaration present: Dockerfile")

    workflow = (ROOT / ".github" / "workflows" / "docker.yml").read_text(encoding="utf-8")
    for required in ["DOCKER_METADATA_SHORT_SHA_LENGTH: 12", "type=sha", "ghcr.io/${{ github.repository }}"]:
        if required not in workflow:
            raise ValueError(f"Docker workflow is missing: {required}")
    print("Immutable 12-character image tags present: .github/workflows/docker.yml")


if __name__ == "__main__":
    validate_python()
    validate_notebooks()
    validate_text_artifacts()
