from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.pull_oci_resumable import (
    blob_is_valid,
    parse_image_reference,
    select_platform_manifest,
)
from voice_tts.core import (
    COSYVOICE3_PROMPT_PREFIX,
    format_prompt_text,
    list_voice_names,
    load_voice_reference,
    sanitize_voice_name,
    validate_tts_text,
)
from voice_tts.library import import_voice_library_zip


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

    def test_voice_name_is_sanitized(self) -> None:
        self.assertEqual(sanitize_voice_name(" max main "), "max_main")
        self.assertEqual(sanitize_voice_name("голос-1"), "голос-1")
        with self.assertRaises(ValueError):
            sanitize_voice_name("../")

    def test_prompt_prefix_is_added_once(self) -> None:
        result = format_prompt_text("Привет, это мой голос")
        self.assertEqual(result, COSYVOICE3_PROMPT_PREFIX + "Привет, это мой голос")
        result = format_prompt_text(COSYVOICE3_PROMPT_PREFIX + " Hello")
        self.assertEqual(result, COSYVOICE3_PROMPT_PREFIX + "Hello")

    def test_text_validation(self) -> None:
        self.assertEqual(validate_tts_text("  Hello   world  "), "Hello world")
        with self.assertRaises(ValueError):
            validate_tts_text("  ")

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
