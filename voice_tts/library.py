from __future__ import annotations

import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from .core import AUDIO_EXTENSIONS, sanitize_voice_name, voices_dir


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = []
    for member in archive.infolist():
        path = PurePosixPath(member.filename.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe ZIP path: {member.filename}")
        unix_mode = member.external_attr >> 16
        if stat.S_ISLNK(unix_mode):
            raise ValueError(f"ZIP symlinks are not allowed: {member.filename}")
        members.append(member)
    return members


def _find_voices_dir(extracted: Path) -> Path:
    direct = extracted / "voices"
    if direct.is_dir():
        return direct
    candidates = sorted(path for path in extracted.rglob("voices") if path.is_dir())
    if len(candidates) != 1:
        raise ValueError("ZIP must contain exactly one voices folder.")
    return candidates[0]


def import_voice_library_zip(zip_path: Path, data_dir: Path, *, overwrite: bool = False) -> dict[str, int]:
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)

    imported_folders = 0
    imported_files = 0
    with tempfile.TemporaryDirectory(prefix="voice_tts_zip_") as temp:
        extracted = Path(temp)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extracted, members=_safe_members(archive))

        source_root = _find_voices_dir(extracted)
        target_root = voices_dir(data_dir)
        for source_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
            voice_name = sanitize_voice_name(source_dir.name)
            target_dir = target_root / voice_name
            target_dir.mkdir(parents=True, exist_ok=True)
            folder_files = 0
            for source in sorted(path for path in source_dir.rglob("*") if path.is_file()):
                if source.suffix.lower() not in AUDIO_EXTENSIONS and source.name != "metadata.json":
                    continue
                relative = source.relative_to(source_dir)
                target = target_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and not overwrite:
                    continue
                shutil.copy2(source, target)
                imported_files += 1
                folder_files += 1
            if folder_files:
                imported_folders += 1

    return {"voice_folders": imported_folders, "files": imported_files}

