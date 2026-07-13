from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

from .core import format_prompt_text


_MODEL = None
_WHISPER = None
_MODEL_LOCK = threading.Lock()
_WHISPER_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


def _truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def cosyvoice_repo() -> Path:
    configured = os.getenv("COSYVOICE_REPO")
    candidates = [
        Path(configured) if configured else None,
        Path("/opt/CosyVoice"),
        Path("/content/CosyVoice"),
    ]
    for candidate in candidates:
        if candidate and (candidate / "cosyvoice").is_dir():
            return candidate.resolve()
    raise RuntimeError("CosyVoice source is missing. Run the Colab bootstrap or build the Docker image.")


def configure_cosyvoice_imports() -> Path:
    repo = cosyvoice_repo()
    for path in [repo, repo / "third_party" / "Matcha-TTS"]:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return repo


def require_cuda() -> None:
    import torch

    if _truthy("REQUIRE_CUDA", "1") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA недоступна. В Colab выберите Runtime -> Change runtime type -> L4 GPU. "
            "Для Docker нужен NVIDIA Container Toolkit и запуск с GPU."
        )


def gpu_summary() -> str:
    import torch

    if not torch.cuda.is_available():
        return "CUDA не обнаружена"
    props = torch.cuda.get_device_properties(0)
    memory_gb = props.total_memory / (1024 ** 3)
    return f"{props.name}, CUDA {torch.version.cuda}, {memory_gb:.1f} GB VRAM"


def model_dir() -> Path:
    return Path(os.getenv("MODEL_DIR", "/models/Fun-CosyVoice3-0.5B")).expanduser().resolve()


def ensure_model_downloaded() -> Path:
    target = model_dir()
    revision = os.getenv("MODEL_REVISION", "29e01c4e8d000f4bcd70751be16fa94bf3d85a18")
    revision_file = target / ".voice_tts_revision"
    if (
        (target / "cosyvoice3.yaml").is_file()
        and revision_file.is_file()
        and revision_file.read_text(encoding="utf-8").strip() == revision
    ):
        return target

    target.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=os.getenv("MODEL_ID", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"),
        revision=revision,
        local_dir=str(target),
    )
    if not (target / "cosyvoice3.yaml").is_file():
        raise RuntimeError(f"Модель скачана не полностью: {target}")
    revision_file.write_text(revision + "\n", encoding="utf-8")
    return target


def get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        require_cuda()
        configure_cosyvoice_imports()
        from cosyvoice.cli.cosyvoice import AutoModel

        _MODEL = AutoModel(
            model_dir=str(ensure_model_downloaded()),
            fp16=_truthy("COSYVOICE_FP16", "1"),
        )
        return _MODEL


def get_whisper_model():
    global _WHISPER
    if _WHISPER is not None:
        return _WHISPER

    with _WHISPER_LOCK:
        if _WHISPER is not None:
            return _WHISPER
        require_cuda()
        import torch
        import whisper

        device = "cuda" if torch.cuda.is_available() else "cpu"
        download_root = Path(os.getenv("WHISPER_CACHE_DIR", "/models/whisper")).expanduser()
        download_root.mkdir(parents=True, exist_ok=True)
        _WHISPER = whisper.load_model(
            os.getenv("WHISPER_MODEL", "base"),
            device=device,
            download_root=str(download_root),
        )
        return _WHISPER


def normalize_reference(source: Path, target: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    probe = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = float(probe.stdout.strip())
    if duration < 3.0:
        raise ValueError("Референс слишком короткий. Запишите хотя бы 3 секунды чистой речи.")
    if duration > 30.0:
        raise ValueError("Референс длиннее 30 секунд. Обрежьте его до 3-30 секунд.")

    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i", str(source),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-af", "loudnorm=I=-20:TP=-1.5:LRA=11",
            str(target),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError("FFmpeg не создал нормализованный WAV.")
    return target


def transcribe_reference(reference_wav: Path) -> str:
    import torch

    model = get_whisper_model()
    result = model.transcribe(
        str(reference_wav),
        fp16=torch.cuda.is_available(),
        task="transcribe",
    )
    transcript = " ".join(str(result.get("text", "")).split())
    if not transcript:
        raise RuntimeError("Whisper не смог распознать речь. Введите расшифровку вручную.")
    return transcript


def synthesize(
    *,
    text: str,
    transcript: str,
    reference_wav: Path,
    output_wav: Path,
    speed: float,
    seed: int,
) -> Path:
    import torch
    import torchaudio

    configure_cosyvoice_imports()
    from cosyvoice.utils.common import set_all_random_seed

    model = get_model()
    prompt_text = format_prompt_text(transcript)
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    with _INFERENCE_LOCK, torch.inference_mode():
        set_all_random_seed(int(seed))
        pieces = []
        for result in model.inference_zero_shot(
            text,
            prompt_text,
            str(reference_wav),
            stream=False,
            speed=float(speed),
        ):
            speech = result["tts_speech"].detach().cpu()
            if speech.ndim == 1:
                speech = speech.unsqueeze(0)
            pieces.append(speech)

        if not pieces:
            raise RuntimeError("CosyVoice не вернул аудио.")
        waveform = torch.cat(pieces, dim=1)
        torchaudio.save(str(output_wav), waveform, model.sample_rate)
    return output_wav
