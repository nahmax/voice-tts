from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
COSYVOICE3_PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"
MAX_TEXT_LENGTH = 1_500


@dataclass(frozen=True)
class VoiceReference:
    name: str
    audio_path: Path
    transcript: str


def sanitize_voice_name(value: str | None) -> str:
    """Return a traversal-safe folder name while keeping Unicode letters."""
    raw = (value or "").strip()
    cleaned = re.sub(r"[^\w-]+", "_", raw, flags=re.UNICODE).strip("_-.")
    cleaned = cleaned[:80]
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError("Укажите имя голоса, например max_main.")
    return cleaned


def validate_tts_text(value: str | None) -> str:
    text = " ".join((value or "").split())
    if not text:
        raise ValueError("Введите текст, который нужно озвучить.")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"Текст слишком длинный: максимум {MAX_TEXT_LENGTH} символов за один запуск.")
    return text


def format_prompt_text(transcript: str | None) -> str:
    raw = (transcript or "").strip()
    if raw.startswith(COSYVOICE3_PROMPT_PREFIX):
        raw = raw[len(COSYVOICE3_PROMPT_PREFIX):]
    clean = " ".join(raw.replace("<|endofprompt|>", " ").split())
    if not clean:
        raise ValueError("Не удалось получить расшифровку референсной записи.")
    return COSYVOICE3_PROMPT_PREFIX + clean


def voices_dir(data_dir: Path) -> Path:
    result = data_dir / "voices"
    result.mkdir(parents=True, exist_ok=True)
    return result


def runs_dir(data_dir: Path) -> Path:
    result = data_dir / "runs"
    result.mkdir(parents=True, exist_ok=True)
    return result


def list_voice_names(data_dir: Path) -> list[str]:
    root = voices_dir(data_dir)
    return sorted(path.name for path in root.iterdir() if path.is_dir() and _first_audio(path) is not None)


def _first_audio(folder: Path) -> Path | None:
    preferred = folder / "reference.wav"
    if preferred.is_file():
        return preferred
    candidates = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )
    return candidates[0] if candidates else None


def load_voice_reference(data_dir: Path, name: str) -> VoiceReference:
    safe_name = sanitize_voice_name(name)
    if safe_name != name:
        raise ValueError("Некорректное имя сохранённого голоса.")

    root = voices_dir(data_dir).resolve()
    folder = (root / safe_name).resolve()
    if root not in folder.parents or not folder.is_dir():
        raise FileNotFoundError(f"Голос не найден: {safe_name}")

    audio_path = _first_audio(folder)
    if audio_path is None:
        raise FileNotFoundError(f"В папке {safe_name} нет поддерживаемого аудиофайла.")

    transcript = ""
    metadata_path = folder / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            transcript = str(metadata.get("transcript", "")).strip()
        except (json.JSONDecodeError, OSError):
            transcript = ""

    return VoiceReference(name=safe_name, audio_path=audio_path, transcript=transcript)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
