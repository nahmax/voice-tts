from __future__ import annotations

import json
import hashlib
import os
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts.pull_oci_resumable import (
    blob_is_valid,
    format_bytes,
    format_duration,
    parse_image_reference,
    select_platform_manifest,
)
from voice_tts.core import (
    COSYVOICE3_PROMPT_PREFIX,
    format_prompt_text,
    list_voice_names,
    load_voice_reference,
    split_tts_text,
    spoken_word_coverage,
    sanitize_voice_name,
    validate_tts_text,
)
from voice_tts.library import import_voice_library_zip
from voice_tts.runtime import (
    COSYVOICE_ENGINE,
    VOXCPM_ENGINE,
    available_tts_engines,
    default_tts_engine,
    normalize_reference,
    normalize_tts_engine,
)


class CoreTests(unittest.TestCase):
    def test_ghcr_reference_and_platform_selection(self) -> None:
        image = parse_image_reference("ghcr.io/NahMax/voice-tts:sha-123456789abc")
        self.assertEqual(image.repository, "nahmax/voice-tts")
        self.assertEqual(image.reference, "sha-123456789abc")
        descriptor = select_platform_manifest(
            {
                "manifests": [
                    {"digest": "sha256:attestation", "platform": {"os": "unknown", "architecture": "unknown"}},
                    {"digest": "sha256:amd64", "platform": {"os": "linux", "architecture": "amd64"}},
                ]
            },
            "linux",
            "amd64",
        )
        self.assertEqual(descriptor["digest"], "sha256:amd64")

    def test_cached_blob_requires_size_and_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "blob"
            payload = b"verified OCI blob"
            path.write_bytes(payload)
            digest = "sha256:" + hashlib.sha256(payload).hexdigest()
            self.assertTrue(blob_is_valid(path, digest, len(payload)))
            self.assertFalse(blob_is_valid(path, digest, len(payload) + 1))

    def test_download_progress_formatting(self) -> None:
        self.assertEqual(format_bytes(1024**3), "1.00 GiB")
        self.assertEqual(format_bytes(64 * 1024**2), "64.00 MiB")
        self.assertEqual(format_duration(65), "1m 05s")
        self.assertEqual(format_duration(3661), "1h 01m 01s")

    def test_voice_name_is_sanitized(self) -> None:
        self.assertEqual(sanitize_voice_name(" max main "), "max_main")
        self.assertEqual(sanitize_voice_name("голос-1"), "голос-1")
        with self.assertRaises(ValueError):
            sanitize_voice_name("../")

    def test_prompt_prefix_is_added_once(self) -> None:
        result = format_prompt_text("Привет, это мой голос")
        self.assertEqual(result, COSYVOICE3_PROMPT_PREFIX + "Привет, это мой голос.")
        result = format_prompt_text(COSYVOICE3_PROMPT_PREFIX + " Hello")
        self.assertEqual(result, COSYVOICE3_PROMPT_PREFIX + "Hello.")
        result = format_prompt_text("Уже завершено!")
        self.assertEqual(result, COSYVOICE3_PROMPT_PREFIX + "Уже завершено!")

    def test_text_validation(self) -> None:
        self.assertEqual(validate_tts_text("  Hello   world  "), "Hello world")
        self.assertEqual(
            validate_tts_text("Первая часть. [orgasm] Вторая часть."),
            "Первая часть. [vocalized-noise] Вторая часть.",
        )
        with self.assertRaisesRegex(ValueError, "Неподдерживаемый тег"):
            validate_tts_text("Текст [unknown_effect] продолжение")
        with self.assertRaises(ValueError):
            validate_tts_text("  ")

    def test_long_text_is_split_without_losing_words(self) -> None:
        text = "Первая фраза полностью. Вторая фраза тоже должна прозвучать. " * 5
        chunks = split_tts_text(text.strip(), max_chars=80)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(" ".join(chunks), text.strip())

    def test_short_sentences_are_never_merged_into_one_model_call(self) -> None:
        text = "Первая фраза должна прозвучать. Вторая тоже не должна пропасть."
        self.assertEqual(
            split_tts_text(text),
            ["Первая фраза должна прозвучать.", "Вторая тоже не должна пропасть."],
        )

    def test_spoken_word_coverage_detects_truncated_output(self) -> None:
        target = "Первая фраза должна прозвучать. [breath] Вторая тоже не должна пропасть."
        self.assertLess(spoken_word_coverage(target, "Первая фраза прозвучала."), 0.72)
        self.assertGreater(
            spoken_word_coverage(target, "Первая фраза должна прозвучать. Вторая тоже не должна пропасть."),
            0.9,
        )

    def test_long_reference_is_cropped_to_thirty_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.wav"
            target = root / "normalized.wav"
            source.write_bytes(b"audio")

            def fake_run(command, **kwargs):
                if command[0] == "ffprobe":
                    return subprocess.CompletedProcess(command, 0, stdout="44.0\n", stderr="")
                target.write_bytes(b"normalized")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with patch("voice_tts.runtime.subprocess.run", side_effect=fake_run) as run_mock:
                self.assertEqual(normalize_reference(source, target), target)

            ffmpeg_command = run_mock.call_args_list[1].args[0]
            duration_index = ffmpeg_command.index("-t") + 1
            self.assertEqual(ffmpeg_command[duration_index], "30.000")

    def test_reference_clip_uses_selected_start_and_duration(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.ogg"
            target = root / "normalized.wav"
            source.write_bytes(b"audio")

            def fake_run(command, **kwargs):
                if command[0] == "ffprobe":
                    return subprocess.CompletedProcess(command, 0, stdout="44.0\n", stderr="")
                target.write_bytes(b"normalized")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with patch("voice_tts.runtime.subprocess.run", side_effect=fake_run) as run_mock:
                normalize_reference(source, target, start_seconds=6, duration_seconds=12)

            ffmpeg_command = run_mock.call_args_list[1].args[0]
            self.assertEqual(ffmpeg_command[ffmpeg_command.index("-ss") + 1], "6.000")
            self.assertEqual(ffmpeg_command[ffmpeg_command.index("-t") + 1], "12.000")

    def test_tts_engine_choices_keep_cosyvoice_as_fallback(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VOXCPM_WORKER_URL", None)
            os.environ.pop("DEFAULT_TTS_ENGINE", None)
            self.assertEqual(available_tts_engines(), [("CosyVoice 3 — проверенный fallback", COSYVOICE_ENGINE)])
            self.assertEqual(default_tts_engine(), COSYVOICE_ENGINE)
            with self.assertRaisesRegex(RuntimeError, "VoxCPM2 не запущен"):
                normalize_tts_engine(VOXCPM_ENGINE)

        with patch.dict(
            os.environ,
            {"VOXCPM_WORKER_URL": "http://127.0.0.1:8765", "DEFAULT_TTS_ENGINE": VOXCPM_ENGINE},
        ):
            self.assertEqual(default_tts_engine(), VOXCPM_ENGINE)
            self.assertEqual(available_tts_engines()[0][1], VOXCPM_ENGINE)
            self.assertEqual(normalize_tts_engine(VOXCPM_ENGINE), VOXCPM_ENGINE)

    def test_unknown_tts_engine_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Неизвестный TTS-движок"):
            normalize_tts_engine("future-model")

    def test_voice_library_discovers_audio_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            voice_dir = root / "voices" / "voice_0"
            voice_dir.mkdir(parents=True)
            (voice_dir / "reference.wav").write_bytes(b"RIFF")
            (voice_dir / "metadata.json").write_text(
                json.dumps({"transcript": "hello"}),
                encoding="utf-8",
            )
            self.assertEqual(list_voice_names(root), ["voice_0"])
            reference = load_voice_reference(root, "voice_0")
            self.assertEqual(reference.transcript, "hello")
            self.assertEqual(reference.audio_path.name, "reference.wav")

    def test_zip_library_import_and_traversal_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            valid_zip = root / "valid.zip"
            with zipfile.ZipFile(valid_zip, "w") as archive:
                archive.writestr("voices/max_main/sample.wav", b"RIFF")
                archive.writestr("voices/max_main/README.txt", b"ignored")
            result = import_voice_library_zip(valid_zip, root / "data")
            self.assertEqual(result, {"voice_folders": 1, "files": 1})
            self.assertTrue((root / "data" / "voices" / "max_main" / "sample.wav").is_file())

            unsafe_zip = root / "unsafe.zip"
            with zipfile.ZipFile(unsafe_zip, "w") as archive:
                archive.writestr("../escape.wav", b"RIFF")
            with self.assertRaises(ValueError):
                import_voice_library_zip(unsafe_zip, root / "data")


if __name__ == "__main__":
    unittest.main()
