from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import gradio as gr

from voice_tts.core import (
    list_voice_names,
    load_voice_reference,
    runs_dir,
    sanitize_voice_name,
    validate_tts_text,
    voices_dir,
    write_json,
)
from voice_tts.runtime import (
    gpu_summary,
    normalize_reference,
    require_cuda,
    synthesize,
    transcribe_reference,
)


DATA_DIR = Path(os.getenv("DATA_DIR", "/data")).expanduser().resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGGER = logging.getLogger("voice_tts")


def _voice_choices() -> list[str]:
    return [""] + list_voice_names(DATA_DIR)


def _new_run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = runs_dir(DATA_DIR) / f"{stamp}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _selected_reference(audio_path: str | None, saved_voice: str | None, run_dir: Path) -> tuple[Path, str, str]:
    transcript = ""
    selected_name = ""

    if audio_path:
        source = Path(audio_path)
    elif saved_voice:
        saved = load_voice_reference(DATA_DIR, saved_voice)
        source = saved.audio_path
        transcript = saved.transcript
        selected_name = saved.name
    else:
        raise ValueError("Загрузите/запишите голос или выберите сохранённый голос.")

    normalized = normalize_reference(source, run_dir / "reference.wav")
    return normalized, transcript, selected_name


def _save_voice(name: str, reference_wav: Path, transcript: str) -> str:
    safe_name = sanitize_voice_name(name)
    target_dir = voices_dir(DATA_DIR) / safe_name
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(reference_wav, target_dir / "reference.wav")
    write_json(
        target_dir / "metadata.json",
        {
            "name": safe_name,
            "transcript": transcript,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return safe_name


def refresh_voice_choices(current: str | None):
    choices = _voice_choices()
    value = current if current in choices else ""
    return gr.Dropdown(choices=choices, value=value)


def transcribe_ui(
    audio_path: str | None,
    saved_voice: str | None,
    consent: bool,
) -> tuple[str, str]:
    try:
        if not consent:
            raise ValueError("Подтвердите право использовать этот голос.")
        with tempfile.TemporaryDirectory(prefix="voice_tts_reference_") as temp_dir:
            reference_wav, saved_transcript, _ = _selected_reference(
                audio_path,
                saved_voice,
                Path(temp_dir),
            )
            transcript = saved_transcript or transcribe_reference(reference_wav)
        return transcript, f"Референс распознан. GPU: {gpu_summary()}"
    except Exception as exc:
        LOGGER.exception("Reference transcription failed")
        return "", f"Ошибка: {exc}"


def generate_ui(
    audio_path: str | None,
    saved_voice: str | None,
    transcript: str | None,
    target_text: str | None,
    voice_name: str | None,
    save_voice: bool,
    speed: float,
    seed: int,
    consent: bool,
):
    try:
        if not consent:
            raise ValueError("Подтвердите, что это ваш голос или у вас есть явное разрешение владельца.")

        text = validate_tts_text(target_text)
        speed_value = float(speed)
        if not 0.75 <= speed_value <= 1.5:
            raise ValueError("Скорость должна быть в диапазоне 0.75-1.5.")

        run_dir = _new_run_dir()
        reference_wav, saved_transcript, selected_name = _selected_reference(audio_path, saved_voice, run_dir)
        final_transcript = " ".join((transcript or saved_transcript).split())
        if not final_transcript:
            final_transcript = transcribe_reference(reference_wav)

        final_voice_name = selected_name
        if audio_path and save_voice:
            requested_name = voice_name or f"voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            final_voice_name = _save_voice(requested_name, reference_wav, final_transcript)

        output_wav = synthesize(
            text=text,
            transcript=final_transcript,
            reference_wav=reference_wav,
            output_wav=run_dir / "output.wav",
            speed=speed_value,
            seed=int(seed),
        )
        write_json(
            run_dir / "run.json",
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "voice_name": final_voice_name,
                "reference_transcript": final_transcript,
                "text": text,
                "speed": speed_value,
                "seed": int(seed),
                "reference_wav": str(reference_wav),
                "output_wav": str(output_wav),
                "gpu": gpu_summary(),
            },
        )

        status = (
            f"Готово. Голос: {final_voice_name or 'временный'}, GPU: {gpu_summary()}. "
            f"Результат сохранён в {run_dir}."
        )
        selected = final_voice_name if final_voice_name else (saved_voice or "")
        return str(output_wav), status, gr.Dropdown(choices=_voice_choices(), value=selected)
    except Exception as exc:
        LOGGER.exception("Speech generation failed")
        return None, f"Ошибка: {exc}", refresh_voice_choices(saved_voice)


CSS = """
.gradio-container { max-width: 1080px !important; }
#primary-action { min-height: 52px; font-weight: 700; }
"""


def create_demo() -> gr.Blocks:
    with gr.Blocks(title="Voice TTS GPU", css=CSS) as demo:
        gr.Markdown(
            """
# Voice TTS GPU

Загрузите аудио или запишите свой голос, введите текст и получите WAV через CosyVoice 3.
Модель загружается один раз при первой генерации. В Colab данные сохраняются на Google Drive.

Используйте только собственный голос или голос, на использование которого у вас есть явное разрешение.
"""
        )

        with gr.Row():
            with gr.Column(scale=3):
                reference_audio = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Новый голосовой референс (3-30 секунд)",
                )
            with gr.Column(scale=2):
                saved_voice = gr.Dropdown(
                    choices=_voice_choices(),
                    value="",
                    label="Или сохранённый голос",
                    info="Новый референс имеет приоритет над сохранённым.",
                )
                refresh_button = gr.Button("Обновить список голосов")

        transcript = gr.Textbox(
            label="Что сказано в референсе",
            lines=2,
            placeholder="Можно оставить пустым: Whisper распознает автоматически.",
        )
        transcribe_button = gr.Button("Распознать референс")

        target_text = gr.Textbox(
            label="Текст для озвучивания",
            lines=5,
            max_length=1500,
            placeholder="Введите короткий текст на русском, английском, немецком или другом поддерживаемом языке.",
        )

        with gr.Row():
            voice_name = gr.Textbox(
                label="Имя нового голоса",
                value="my_voice",
                info="Используется только при загрузке/записи нового референса.",
            )
            save_voice = gr.Checkbox(value=True, label="Сохранить новый голос в библиотеке")

        with gr.Accordion("Дополнительные настройки", open=False):
            speed = gr.Slider(0.75, 1.5, value=1.0, step=0.05, label="Скорость")
            seed = gr.Number(value=42, precision=0, label="Seed")

        consent = gr.Checkbox(
            value=False,
            label="Я подтверждаю, что это мой голос или у меня есть явное разрешение владельца",
        )
        generate_button = gr.Button("Сгенерировать речь", variant="primary", elem_id="primary-action")

        output_audio = gr.Audio(label="Результат", type="filepath", autoplay=False)
        status = gr.Markdown("Ожидание запуска.")

        refresh_button.click(refresh_voice_choices, inputs=[saved_voice], outputs=[saved_voice])
        transcribe_button.click(
            transcribe_ui,
            inputs=[reference_audio, saved_voice, consent],
            outputs=[transcript, status],
        )
        generate_button.click(
            generate_ui,
            inputs=[
                reference_audio,
                saved_voice,
                transcript,
                target_text,
                voice_name,
                save_voice,
                speed,
                seed,
                consent,
            ],
            outputs=[output_audio, status, saved_voice],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice TTS Gradio application")
    parser.add_argument("--host", default=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("GRADIO_SERVER_PORT", "7860")))
    parser.add_argument("--share", action="store_true", default=os.getenv("GRADIO_SHARE", "0") == "1")
    args = parser.parse_args()
    username = os.getenv("GRADIO_USERNAME", "").strip()
    password = os.getenv("GRADIO_PASSWORD", "").strip()
    auth = (username, password) if username and password else None
    if password == "change-me-now":
        raise RuntimeError("Replace the default GRADIO_PASSWORD before starting the service.")
    if args.share and auth is None:
        raise RuntimeError("A public Gradio share requires GRADIO_USERNAME and GRADIO_PASSWORD.")
    require_cuda()
    print("GPU:", gpu_summary(), flush=True)

    demo = create_demo()
    demo.queue(max_size=4, default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        auth=auth,
        show_error=True,
    )


if __name__ == "__main__":
    main()
