from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
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
    split_tts_text,
    spoken_word_coverage,
    validate_tts_text,
    voices_dir,
    write_json,
)
from voice_tts.runtime import (
    available_tts_engines,
    default_tts_engine,
    engine_display_name,
    gpu_summary,
    normalize_reference,
    normalize_tts_engine,
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


def _create_browser_preview(output_wav: Path) -> Path:
    """Create a small MP3 for the web player while preserving the 48 kHz WAV."""
    preview_path = output_wav.with_name("preview.mp3")
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(output_wav),
            "-vn",
            "-ac",
            "1",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "96k",
            str(preview_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not preview_path.is_file() or preview_path.stat().st_size == 0:
        LOGGER.warning("MP3 preview creation failed: %s", completed.stderr[-1000:])
        return output_wav
    return preview_path


def _selected_reference(
    audio_path: str | None,
    saved_voice: str | None,
    run_dir: Path,
    clip_start: float = 0.0,
    clip_duration: float = 30.0,
) -> tuple[Path, str, str]:
    transcript = ""
    selected_name = ""

    if audio_path:
        source = Path(audio_path)
        start_seconds = float(clip_start)
        duration_seconds = float(clip_duration)
    elif saved_voice:
        saved = load_voice_reference(DATA_DIR, saved_voice)
        source = saved.audio_path
        transcript = saved.transcript
        selected_name = saved.name
        start_seconds = 0.0
        duration_seconds = 30.0
    else:
        raise ValueError("Загрузите/запишите голос или выберите сохранённый голос.")

    normalized = normalize_reference(
        source,
        run_dir / "reference.wav",
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
    )
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
    clip_start: float,
    clip_duration: float,
    consent: bool,
    progress=gr.Progress(),
) -> tuple[str, str]:
    try:
        progress(0.05, desc="Проверка референса")
        if not consent:
            raise ValueError("Подтвердите право использовать этот голос.")
        with tempfile.TemporaryDirectory(prefix="voice_tts_reference_") as temp_dir:
            progress(0.20, desc="Подготовка выбранного отрезка")
            reference_wav, saved_transcript, selected_name = _selected_reference(
                audio_path,
                saved_voice,
                Path(temp_dir),
                clip_start,
                clip_duration,
            )
            progress(0.45, desc="Точное распознавание речи (Whisper medium)")
            transcript = saved_transcript or transcribe_reference(reference_wav)
            if selected_name and not saved_transcript:
                _save_voice(selected_name, reference_wav, transcript)
        progress(1.0, desc="Референс готов")
        clip_note = f" Использован отрезок {float(clip_start):g}–{float(clip_start) + float(clip_duration):g} с." if audio_path else ""
        return transcript, f"Референс распознан.{clip_note} GPU: {gpu_summary()}"
    except Exception as exc:
        LOGGER.exception("Reference transcription failed")
        return "", f"Ошибка: {exc}"


def generate_ui(
    engine: str,
    audio_path: str | None,
    saved_voice: str | None,
    clip_start: float,
    clip_duration: float,
    transcript: str | None,
    target_text: str | None,
    voice_name: str | None,
    save_voice: bool,
    speed: float,
    seed: int,
    consent: bool,
    progress=gr.Progress(),
):
    try:
        progress(0.03, desc="Проверка параметров")
        if not consent:
            raise ValueError("Подтвердите, что это ваш голос или у вас есть явное разрешение владельца.")

        text = validate_tts_text(target_text)
        selected_engine = normalize_tts_engine(engine)
        speed_value = float(speed)
        if not 0.75 <= speed_value <= 1.5:
            raise ValueError("Скорость должна быть в диапазоне 0.75-1.5.")

        progress(0.10, desc="Подготовка голосового референса")
        run_dir = _new_run_dir()
        reference_wav, saved_transcript, selected_name = _selected_reference(
            audio_path,
            saved_voice,
            run_dir,
            clip_start,
            clip_duration,
        )
        # A saved voice must always use its own aligned transcript. Otherwise a
        # stale textbox value from a previously uploaded file corrupts cloning.
        transcript_source = transcript if audio_path else (saved_transcript or transcript)
        final_transcript = " ".join((transcript_source or "").split())
        if not final_transcript:
            progress(0.25, desc="Распознавание референса (Whisper medium)")
            final_transcript = transcribe_reference(reference_wav)

        final_voice_name = selected_name
        if audio_path and save_voice:
            requested_name = voice_name or f"voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            final_voice_name = _save_voice(requested_name, reference_wav, final_transcript)

        engine_name = engine_display_name(selected_engine)
        progress(0.50, desc=f"Загрузка {engine_name} и подготовка генерации")

        def report_synthesis(done: int, total: int) -> None:
            fraction = done / max(total, 1)
            progress(
                0.55 + 0.33 * fraction,
                desc=f"Озвучивание: часть {min(done + 1, total)} из {total}" if done < total else "Сборка WAV",
            )

        output_wav = synthesize(
            engine=selected_engine,
            text=text,
            transcript=final_transcript,
            reference_wav=reference_wav,
            output_wav=run_dir / "output.wav",
            speed=speed_value,
            seed=int(seed),
            progress_callback=report_synthesis,
        )
        progress(0.90, desc="Проверка, что произнесён весь текст")
        output_transcript = transcribe_reference(output_wav)
        output_coverage = spoken_word_coverage(text, output_transcript)
        if output_coverage < 0.72:
            raise RuntimeError(
                f"Модель произнесла только около {output_coverage:.0%} текста. "
                "Выберите чистый референс 8–15 секунд и снова нажмите «Подготовить и распознать голос»."
            )

        progress(0.97, desc="Сохранение проверенного результата")
        preview_audio = _create_browser_preview(output_wav)
        write_json(
            run_dir / "run.json",
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "tts_engine": selected_engine,
                "tts_engine_name": engine_name,
                "voice_name": final_voice_name,
                "reference_transcript": final_transcript,
                "text": text,
                "speed": speed_value,
                "seed": int(seed),
                "reference_wav": str(reference_wav),
                "reference_clip_start": float(clip_start) if audio_path else 0.0,
                "reference_clip_duration": float(clip_duration) if audio_path else 30.0,
                "output_wav": str(output_wav),
                "preview_audio": str(preview_audio),
                "output_transcript": output_transcript,
                "spoken_word_coverage": output_coverage,
                "gpu": gpu_summary(),
            },
        )

        chunk_count = len(split_tts_text(text))
        status = (
            f"Готово через {engine_name}: озвучено частей — {chunk_count}, "
            f"полнота проверена ({output_coverage:.0%}). "
            f"Голос: {final_voice_name or 'временный'}, GPU: {gpu_summary()}. "
            f"Результат сохранён в {run_dir}."
        )
        progress(1.0, desc="Готово")
        selected = final_voice_name if final_voice_name else (saved_voice or "")
        return (
            str(preview_audio),
            str(output_wav),
            status,
            gr.Dropdown(choices=_voice_choices(), value=selected),
        )
    except Exception as exc:
        LOGGER.exception("Speech generation failed")
        return None, None, f"Ошибка: {exc}", refresh_voice_choices(saved_voice)


CSS = """
.gradio-container { max-width: 1080px !important; }
#primary-action { min-height: 52px; font-weight: 700; }
"""


def create_demo() -> gr.Blocks:
    with gr.Blocks(title="Voice TTS GPU", css=CSS) as demo:
        gr.Markdown(
            """
# Voice TTS GPU

Загрузите аудио или запишите свой голос, выберите VoxCPM2 либо CosyVoice 3 и получите WAV своим голосом.
VoxCPM2 создаёт 48-кГц аудио и используется как основной движок в Colab; CosyVoice 3 остаётся проверенным fallback. Каждое предложение генерируется отдельно, поэтому длинный текст не обрывается. Выбранная модель загружается при первой генерации, а данные в Colab сохраняются на Google Drive.

Используйте только собственный голос или голос, на использование которого у вас есть явное разрешение.
"""
        )

        engine = gr.Radio(
            choices=available_tts_engines(),
            value=default_tts_engine(),
            label="TTS-модель",
            info="VoxCPM2 — основной 48-кГц движок; CosyVoice 3 — стабильный fallback.",
        )

        with gr.Row():
            with gr.Column(scale=3):
                reference_audio = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Исходная запись голоса",
                )
                gr.Markdown(
                    "Выберите чистый отрезок без музыки и перебиваний. Оптимально 8–15 секунд; "
                    "это референс голоса, а не обучение модели. До 30 секунд разрешено, но длиннее не означает лучше."
                )
                with gr.Row():
                    clip_start = gr.Number(
                        value=0,
                        minimum=0,
                        step=0.5,
                        label="Начало отрезка, сек",
                    )
                    clip_duration = gr.Slider(
                        minimum=3,
                        maximum=30,
                        value=10,
                        step=0.5,
                        label="Длина отрезка, сек",
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
        transcribe_button = gr.Button("Подготовить и распознать голос")

        target_text = gr.Textbox(
            label="Текст для озвучивания",
            lines=5,
            max_length=1500,
            placeholder="Введите короткий текст на русском, английском, немецком или другом поддерживаемом языке.",
        )
        gr.Markdown(
            "Для CosyVoice поддерживаются эффекты: `[breath]`, `[quick_breath]`, `[sigh]`, `[cough]`, "
            "`[lipsmack]`, `[mn]`, `[laughter]`, `[noise]`, `[vocalized-noise]`. "
            "Теги `[orgasm]` и `[moan]` автоматически заменяются на `[vocalized-noise]`, "
            "чтобы не обрывать оставшийся текст. VoxCPM2 безопасно пропускает эти CosyVoice-теги."
        )

        with gr.Row():
            voice_name = gr.Textbox(
                label="Имя нового голоса",
                value="my_voice",
                info="Используется только при загрузке/записи нового референса.",
            )
            save_voice = gr.Checkbox(value=True, label="Сохранить новый голос в библиотеке")

        with gr.Accordion("Дополнительные настройки", open=False):
            speed = gr.Slider(
                0.75,
                1.5,
                value=1.0,
                step=0.05,
                label="Скорость",
                info="CosyVoice управляет темпом во время синтеза; для VoxCPM итоговый WAV меняет темп через FFmpeg.",
            )
            seed = gr.Number(value=42, precision=0, label="Seed")

        consent = gr.Checkbox(
            value=False,
            label="Я подтверждаю, что это мой голос или у меня есть явное разрешение владельца",
        )
        generate_button = gr.Button("Сгенерировать речь", variant="primary", elem_id="primary-action")

        output_audio = gr.Audio(
            label="Быстрое превью (MP3)",
            type="filepath",
            autoplay=False,
            format="mp3",
        )
        output_wav_file = gr.File(
            label="Исходный WAV 48 кГц",
            type="filepath",
            interactive=False,
        )
        status = gr.Markdown("Ожидание запуска.")

        refresh_button.click(refresh_voice_choices, inputs=[saved_voice], outputs=[saved_voice])
        transcribe_button.click(
            transcribe_ui,
            inputs=[reference_audio, saved_voice, clip_start, clip_duration, consent],
            outputs=[transcript, status],
        )
        generate_button.click(
            generate_ui,
            inputs=[
                engine,
                reference_audio,
                saved_voice,
                clip_start,
                clip_duration,
                transcript,
                target_text,
                voice_name,
                save_voice,
                speed,
                seed,
                consent,
            ],
            outputs=[output_audio, output_wav_file, status, saved_voice],
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
        allowed_paths=[str(runs_dir(DATA_DIR))],
        show_error=True,
    )


if __name__ == "__main__":
    main()
