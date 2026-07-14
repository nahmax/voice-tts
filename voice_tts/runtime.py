from __future__ import annotations

import base64
import gc
import json
import os
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from .core import format_prompt_text, split_tts_text


_MODEL = None
_WHISPER = None
_MODEL_LOCK = threading.Lock()
_WHISPER_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()

MIN_REFERENCE_SECONDS = 3.0
MAX_REFERENCE_SECONDS = 30.0
COSYVOICE_ENGINE = "cosyvoice3"
VOXCPM_ENGINE = "voxcpm2"
ENGINE_LABELS = {
    VOXCPM_ENGINE: "VoxCPM2 2B — 48 кГц, основной",
    COSYVOICE_ENGINE: "CosyVoice 3 — проверенный fallback",
}


def _truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def voxcpm_worker_url() -> str:
    return os.getenv("VOXCPM_WORKER_URL", "").strip().rstrip("/")


def available_tts_engines() -> list[tuple[str, str]]:
    engines: list[tuple[str, str]] = []
    if voxcpm_worker_url():
        engines.append((ENGINE_LABELS[VOXCPM_ENGINE], VOXCPM_ENGINE))
    engines.append((ENGINE_LABELS[COSYVOICE_ENGINE], COSYVOICE_ENGINE))
    return engines


def normalize_tts_engine(value: str | None) -> str:
    engine = (value or "").strip().lower()
    if not engine:
        engine = COSYVOICE_ENGINE
    if engine not in ENGINE_LABELS:
        raise ValueError(f"Неизвестный TTS-движок: {value}.")
    if engine == VOXCPM_ENGINE and not voxcpm_worker_url():
        raise RuntimeError(
            "VoxCPM2 не запущен. Выполните ячейку подготовки Colab или выберите CosyVoice 3."
        )
    return engine


def default_tts_engine() -> str:
    configured = os.getenv("DEFAULT_TTS_ENGINE", COSYVOICE_ENGINE).strip().lower()
    if configured == VOXCPM_ENGINE and voxcpm_worker_url():
        return VOXCPM_ENGINE
    return COSYVOICE_ENGINE


def engine_display_name(engine: str) -> str:
    return ENGINE_LABELS[normalize_tts_engine(engine)].split(" — ", 1)[0]


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


def release_cosyvoice_model() -> None:
    global _MODEL
    with _MODEL_LOCK:
        model = _MODEL
        _MODEL = None
    if model is None:
        return
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _voxcpm_request(
    path: str,
    payload: dict[str, object] | None = None,
    *,
    timeout: float = 900.0,
) -> dict[str, object]:
    base_url = voxcpm_worker_url()
    if not base_url:
        raise RuntimeError(
            "VoxCPM2 worker не настроен. Выполните ячейку подготовки Colab или выберите CosyVoice 3."
        )

    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    token = os.getenv("VOXCPM_WORKER_TOKEN", "").strip()
    if token:
        headers["X-Voice-TTS-Token"] = token
    request = urllib.request.Request(
        f"{base_url}/{path.lstrip('/')}",
        data=data,
        headers=headers,
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        try:
            message = str(json.loads(details).get("error", details))
        except json.JSONDecodeError:
            message = details
        raise RuntimeError(f"VoxCPM2 worker вернул HTTP {exc.code}: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"VoxCPM2 worker недоступен по адресу {base_url}: {exc.reason}. "
            "Повторно выполните ячейку подготовки Colab."
        ) from exc

    try:
        result = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("VoxCPM2 worker вернул повреждённый JSON.") from exc
    if not isinstance(result, dict):
        raise RuntimeError("VoxCPM2 worker вернул неожиданный ответ.")
    if result.get("ok") is False:
        raise RuntimeError(f"VoxCPM2: {result.get('error', 'неизвестная ошибка')}")
    return result


def voxcpm_health() -> dict[str, object]:
    return _voxcpm_request("health", timeout=5.0)


def release_voxcpm_model() -> None:
    if not voxcpm_worker_url():
        return
    try:
        _voxcpm_request("unload", {}, timeout=60.0)
    except RuntimeError:
        # CosyVoice remains a usable fallback even if an optional worker died.
        return


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
            os.getenv("WHISPER_MODEL", "medium"),
            device=device,
            download_root=str(download_root),
        )
        return _WHISPER


def normalize_reference(
    source: Path,
    target: Path,
    *,
    start_seconds: float = 0.0,
    duration_seconds: float = MAX_REFERENCE_SECONDS,
) -> Path:
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
    source_duration = float(probe.stdout.strip())
    if source_duration < MIN_REFERENCE_SECONDS:
        raise ValueError("Референс слишком короткий. Запишите хотя бы 3 секунды чистой речи.")

    clip_start = float(start_seconds)
    requested_duration = float(duration_seconds)
    if clip_start < 0:
        raise ValueError("Начало референса не может быть отрицательным.")
    if not MIN_REFERENCE_SECONDS <= requested_duration <= MAX_REFERENCE_SECONDS:
        raise ValueError("Длина выбранного референса должна быть от 3 до 30 секунд.")

    available_duration = source_duration - clip_start
    if available_duration < MIN_REFERENCE_SECONDS:
        raise ValueError(
            f"После отметки {clip_start:g} с осталось меньше 3 секунд. Уменьшите начало отрезка."
        )
    output_duration = min(requested_duration, available_duration, MAX_REFERENCE_SECONDS)

    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i", str(source),
            "-ss", f"{clip_start:.3f}",
            "-t", f"{output_duration:.3f}",
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
    engine: str,
    text: str,
    transcript: str,
    reference_wav: Path,
    output_wav: Path,
    speed: float,
    seed: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    selected_engine = normalize_tts_engine(engine)
    with _INFERENCE_LOCK:
        if selected_engine == VOXCPM_ENGINE:
            release_cosyvoice_model()
            return _synthesize_voxcpm(
                text=text,
                transcript=transcript,
                reference_wav=reference_wav,
                output_wav=output_wav,
                speed=speed,
                seed=seed,
                progress_callback=progress_callback,
            )

        release_voxcpm_model()
        return _synthesize_cosyvoice(
            text=text,
            transcript=transcript,
            reference_wav=reference_wav,
            output_wav=output_wav,
            speed=speed,
            seed=seed,
            progress_callback=progress_callback,
        )


def _synthesize_cosyvoice(
    *,
    text: str,
    transcript: str,
    reference_wav: Path,
    output_wav: Path,
    speed: float,
    seed: int,
    progress_callback: Callable[[int, int], None] | None,
) -> Path:
    import torch
    import torchaudio

    configure_cosyvoice_imports()
    from cosyvoice.utils.common import set_all_random_seed

    model = get_model()
    prompt_text = format_prompt_text(transcript)
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        set_all_random_seed(int(seed))
        pieces = []
        text_chunks = split_tts_text(text)
        for chunk_index, text_chunk in enumerate(text_chunks):
            if progress_callback is not None:
                progress_callback(chunk_index, len(text_chunks))
            chunk_pieces = []
            for result in model.inference_zero_shot(
                text_chunk,
                prompt_text,
                str(reference_wav),
                stream=False,
                speed=float(speed),
            ):
                speech = result["tts_speech"].detach().cpu()
                if speech.ndim == 1:
                    speech = speech.unsqueeze(0)
                chunk_pieces.append(speech)
            if not chunk_pieces:
                raise RuntimeError(f"CosyVoice не вернул аудио для части {chunk_index + 1} из {len(text_chunks)}.")
            pieces.extend(chunk_pieces)
            if chunk_index < len(text_chunks) - 1:
                pieces.append(torch.zeros((1, int(model.sample_rate * 0.18))))

        if progress_callback is not None:
            progress_callback(len(text_chunks), len(text_chunks))

        if not pieces:
            raise RuntimeError("CosyVoice не вернул аудио.")
        waveform = torch.cat(pieces, dim=1)
        torchaudio.save(str(output_wav), waveform, model.sample_rate)
    return output_wav


def _synthesize_voxcpm(
    *,
    text: str,
    transcript: str,
    reference_wav: Path,
    output_wav: Path,
    speed: float,
    seed: int,
    progress_callback: Callable[[int, int], None] | None,
) -> Path:
    import re

    import torch
    import torchaudio

    health = voxcpm_health()
    status = str(health.get("status", "unknown"))
    if status == "error":
        raise RuntimeError(f"VoxCPM2 worker не загрузился: {health.get('error', 'unknown error')}")

    reference_base64 = base64.b64encode(reference_wav.read_bytes()).decode("ascii")
    text_chunks = split_tts_text(text)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    pieces = []
    sample_rate: int | None = None

    with tempfile.TemporaryDirectory(prefix="voice_tts_voxcpm_") as temp_dir:
        temp_root = Path(temp_dir)
        for chunk_index, original_chunk in enumerate(text_chunks):
            if progress_callback is not None:
                progress_callback(chunk_index, len(text_chunks))

            # CosyVoice effect tags are not VoxCPM control syntax. Removing them
            # prevents the bracketed tag names from being spoken literally.
            text_chunk = " ".join(re.sub(r"\[[^\]]+\]", " ", original_chunk).split())
            if not text_chunk:
                continue
            response = _voxcpm_request(
                "synthesize",
                {
                    "text": text_chunk,
                    "prompt_text": transcript,
                    "reference_wav_base64": reference_base64,
                    "seed": int(seed) + chunk_index,
                    "cfg_value": float(os.getenv("VOXCPM_CFG_VALUE", "2.0")),
                    "inference_timesteps": int(os.getenv("VOXCPM_INFERENCE_TIMESTEPS", "10")),
                },
            )
            encoded_audio = response.get("wav_base64")
            if not isinstance(encoded_audio, str) or not encoded_audio:
                raise RuntimeError(
                    f"VoxCPM2 не вернул аудио для части {chunk_index + 1} из {len(text_chunks)}."
                )
            try:
                wav_bytes = base64.b64decode(encoded_audio, validate=True)
            except ValueError as exc:
                raise RuntimeError("VoxCPM2 worker вернул повреждённый WAV.") from exc

            chunk_path = temp_root / f"chunk_{chunk_index:03d}.wav"
            chunk_path.write_bytes(wav_bytes)
            waveform, chunk_sample_rate = torchaudio.load(str(chunk_path))
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sample_rate is None:
                sample_rate = int(chunk_sample_rate)
            elif int(chunk_sample_rate) != sample_rate:
                waveform = torchaudio.functional.resample(waveform, int(chunk_sample_rate), sample_rate)
            pieces.append(waveform.cpu())
            if chunk_index < len(text_chunks) - 1:
                pieces.append(torch.zeros((1, int(sample_rate * 0.18))))

        if progress_callback is not None:
            progress_callback(len(text_chunks), len(text_chunks))
        if not pieces or sample_rate is None:
            raise RuntimeError("VoxCPM2 не вернул аудио.")

        waveform = torch.cat(pieces, dim=1)
        if abs(float(speed) - 1.0) < 0.001:
            torchaudio.save(str(output_wav), waveform, sample_rate)
        else:
            raw_path = temp_root / "assembled.wav"
            torchaudio.save(str(raw_path), waveform, sample_rate)
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-i", str(raw_path),
                    "-filter:a", f"atempo={float(speed):.3f}",
                    str(output_wav),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
    return output_wav
