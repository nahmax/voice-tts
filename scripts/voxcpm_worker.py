from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import os
import tempfile
import threading
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_TEXT_LENGTH = 2_000
_MODEL = None
_MODEL_LOCK = threading.Lock()
_STATE_LOCK = threading.Lock()
_STATE: dict[str, Any] = {"status": "idle", "error": ""}


def _set_state(status: str, error: str = "") -> None:
    with _STATE_LOCK:
        _STATE["status"] = status
        _STATE["error"] = error


def state_snapshot() -> dict[str, Any]:
    with _STATE_LOCK:
        return {
            "ok": _STATE["status"] != "error",
            "status": _STATE["status"],
            "error": _STATE["error"],
            "model_id": os.getenv("VOXCPM_MODEL_ID", "openbmb/VoxCPM2"),
        }


def get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        _set_state("loading")
        try:
            import torch
            from voxcpm import VoxCPM

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA недоступна в отдельном VoxCPM2-окружении.")
            model_id = os.getenv("VOXCPM_MODEL_ID", "openbmb/VoxCPM2")
            print(f"Loading VoxCPM2 from {model_id} on {torch.cuda.get_device_name(0)}", flush=True)
            _MODEL = VoxCPM.from_pretrained(model_id, load_denoiser=False)
            _set_state("ready")
            print("VoxCPM2 is ready", flush=True)
            return _MODEL
        except Exception as exc:
            _set_state("error", str(exc))
            traceback.print_exc()
            raise


def unload_model() -> None:
    global _MODEL
    with _MODEL_LOCK:
        model = _MODEL
        _MODEL = None
        _set_state("idle")
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
    print("VoxCPM2 unloaded from GPU", flush=True)


def synthesize(payload: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    import torch

    text = " ".join(str(payload.get("text", "")).split())
    prompt_text = " ".join(str(payload.get("prompt_text", "")).split())
    encoded_reference = payload.get("reference_wav_base64")
    if not text:
        raise ValueError("Пустой текст для VoxCPM2.")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"Одна часть текста длиннее {MAX_TEXT_LENGTH} символов.")
    if not prompt_text:
        raise ValueError("Для точного клонирования VoxCPM2 нужна расшифровка референса.")
    if not isinstance(encoded_reference, str) or not encoded_reference:
        raise ValueError("Не передан голосовой референс.")
    try:
        reference_bytes = base64.b64decode(encoded_reference, validate=True)
    except ValueError as exc:
        raise ValueError("Голосовой референс повреждён.") from exc
    if len(reference_bytes) > 4 * 1024 * 1024:
        raise ValueError("Голосовой референс слишком большой.")

    model = get_model()
    with tempfile.TemporaryDirectory(prefix="voxcpm_worker_") as temp_dir:
        reference_path = Path(temp_dir) / "reference.wav"
        reference_path.write_bytes(reference_bytes)
        with torch.inference_mode():
            wav = model.generate(
                text=text,
                prompt_wav_path=str(reference_path),
                prompt_text=prompt_text,
                reference_wav_path=str(reference_path),
                cfg_value=float(payload.get("cfg_value", 2.0)),
                inference_timesteps=int(payload.get("inference_timesteps", 10)),
                seed=int(payload.get("seed", 42)),
            )

    waveform = np.asarray(wav, dtype=np.float32).squeeze()
    if waveform.ndim != 1 or waveform.size == 0:
        raise RuntimeError(f"VoxCPM2 вернул неожиданный массив аудио: shape={waveform.shape}.")
    sample_rate = int(model.tts_model.sample_rate)
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")
    return {
        "ok": True,
        "sample_rate": sample_rate,
        "wav_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


class WorkerHandler(BaseHTTPRequestHandler):
    server_version = "VoiceTTSVoxCPM/1.0"

    def _authorized(self) -> bool:
        expected = os.getenv("VOXCPM_WORKER_TOKEN", "").strip()
        return not expected or self.headers.get("X-Voice-TTS-Token", "") == expected

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _reject_unauthorized(self) -> bool:
        if self._authorized():
            return False
        self._send_json(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "unauthorized"})
        return True

    def do_GET(self) -> None:  # noqa: N802
        if self._reject_unauthorized():
            return
        if self.path.rstrip("/") == "/health":
            self._send_json(HTTPStatus.OK, state_snapshot())
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self._reject_unauthorized():
            return
        length_text = self.headers.get("Content-Length", "0")
        try:
            length = int(length_text)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"ok": False, "error": "invalid request size"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            if self.path.rstrip("/") == "/synthesize":
                result = synthesize(payload)
            elif self.path.rstrip("/") == "/unload":
                unload_model()
                result = {"ok": True, "status": "idle"}
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
                return
            self._send_json(HTTPStatus.OK, result)
        except Exception as exc:
            traceback.print_exc()
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})

    def log_message(self, message: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {message % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local VoxCPM2 GPU worker for Voice TTS")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--preload", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), WorkerHandler)
    print(f"VoxCPM2 worker listening on http://{args.host}:{args.port}", flush=True)
    if args.preload:
        threading.Thread(target=get_model, name="voxcpm-preload", daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        unload_model()


if __name__ == "__main__":
    main()
