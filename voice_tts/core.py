from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
COSYVOICE3_PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"
MAX_TEXT_LENGTH = 1_500
SUPPORTED_EXPRESSIVE_TAGS = frozenset({
    "[accent]",
    "[breath]",
    "[clucking]",
    "[cough]",
    "[hissing]",
    "[laughter]",
    "[lipsmack]",
    "[mn]",
    "[noise]",
    "[quick_breath]",
    "[sigh]",
    "[vocalized-noise]",
})
EXPRESSIVE_TAG_ALIASES = {
    "[moan]": "[vocalized-noise]",
    "[orgasm]": "[vocalized-noise]",
}
_EXPRESSIVE_TAG_RE = re.compile(r"\[[A-Za-z][A-Za-z0-9_-]{1,31}\]")
_SPOKEN_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


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
    return normalize_tts_tags(text)


def normalize_tts_tags(text: str) -> str:
    """Normalize expressive tags and reject tags that can truncate CosyVoice output."""
    unknown: list[str] = []

    def replace(match: re.Match[str]) -> str:
        tag = match.group(0).lower()
        if tag in EXPRESSIVE_TAG_ALIASES:
            return EXPRESSIVE_TAG_ALIASES[tag]
        if tag in SUPPORTED_EXPRESSIVE_TAGS:
            return tag
        unknown.append(match.group(0))
        return match.group(0)

    normalized = _EXPRESSIVE_TAG_RE.sub(replace, text)
    if unknown:
        supported = ", ".join(sorted(SUPPORTED_EXPRESSIVE_TAGS))
        raise ValueError(
            f"Неподдерживаемый тег: {unknown[0]}. Он может оборвать оставшийся текст. "
            f"Используйте один из тегов: {supported}."
        )
    return normalized


def split_tts_text(text: str, max_chars: int = 180) -> list[str]:
    """Split text into sentence-sized synthesis calls without losing words.

    CosyVoice may stop after the first sentence when several sentences are sent in
    one zero-shot request.  Sentence boundaries are therefore hard boundaries; a
    single unusually long sentence is additionally split on words.
    """
    if max_chars < 40:
        raise ValueError("max_chars must be at least 40")

    sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", text) if part.strip()]
    chunks: list[str] = []
    for sentence in sentences:
        current = ""
        words = sentence.split()
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = word
            else:
                current = candidate
        if current:
            chunks.append(current)
    return chunks or [text]


def format_prompt_text(transcript: str | None) -> str:
    raw = (transcript or "").strip()
    if raw.startswith(COSYVOICE3_PROMPT_PREFIX):
        raw = raw[len(COSYVOICE3_PROMPT_PREFIX):]
    clean = " ".join(raw.replace("<|endofprompt|>", " ").split())
    if not clean:
        raise ValueError("Не удалось получить расшифровку референсной записи.")
    if clean[-1] not in ".!?。！？":
        clean += "."
    return COSYVOICE3_PROMPT_PREFIX + clean


def spoken_word_coverage(target_text: str, recognized_text: str) -> float:
    """Estimate how much target speech exists in an ASR transcript.

    ASR may miss punctuation and proper-name spelling, so truncation detection is
    intentionally based on spoken word counts instead of exact string equality.
    """
    target_without_tags = _EXPRESSIVE_TAG_RE.sub(" ", target_text.lower())
    target_words = _SPOKEN_WORD_RE.findall(target_without_tags)
    recognized_words = _SPOKEN_WORD_RE.findall(recognized_text.lower())
    if not target_words:
        return 1.0
    return min(1.0, len(recognized_words) / len(target_words))


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
