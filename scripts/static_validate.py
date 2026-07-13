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
    for required in [
        'EXECUTION_MODE = "udocker"',
        'UDOCKER_VERSION = "1.3.17"',
        'f"{image_repository}:sha-{commit[:12]}"',
        '"--nvidia"',
        '"--allow-root"',
        'EXECUTION_MODE == "udocker"',
        'EXECUTION_MODE == "native"',
        'require_cuda(); print(ensure_model_downloaded())',
        '"--share", "--port", "7860"',
    ]:
        if required not in code:
            raise ValueError(f"Canonical notebook is missing: {required}")
    print("Colab OCI/GPU workflow present: voice_tts_colab_gpu.ipynb")


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
    for required in ["FROM nvidia/cuda:", "REQUIRE_CUDA=1", "COSYVOICE_FP16=1"]:
        if required not in dockerfile:
            raise ValueError(f"Dockerfile is missing: {required}")
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
