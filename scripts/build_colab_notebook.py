from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "jupyter-notebook" / "voice_tts_colab_gpu.ipynb"


def _source(text: str) -> list[str]:
    normalized = dedent(text).strip("\n") + "\n"
    return normalized.splitlines(keepends=True)


def markdown(text: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source(text),
    }


def code(text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


CELLS = [
    markdown(
        """
        # Voice TTS: единый центр управления в Google Colab L4

        Этот notebook получает код из GitHub, пытается подключить Google Drive и запускает один и тот же Gradio-интерфейс двумя способами:

        - `udocker` (по умолчанию) — исполняет **точный OCI/Docker-образ**, который GitHub Actions собрал для выбранного Git-коммита;
        - `native` — резервный запуск того же `app.py` в изолированном Python 3.10 окружении Colab.

        Через UI можно загрузить или записать разрешённый голос, ввести текст и получить WAV. При рабочем Drive голоса, модели и результаты сохраняются в `MyDrive/Voice TTS/`; при сбое DriveFS notebook явно переключается на временный `/content/voice-tts-data` и не блокирует GPU workflow.
        """
    ),
    markdown(
        """
        ## Важное ограничение и безопасность

        Hosted Colab не предоставляет поддерживаемый Docker daemon с NVIDIA Container Toolkit. Режим `udocker` не запускает Docker Engine: он извлекает доверенный OCI-образ в пользовательское пространство и добавляет в него NVIDIA-библиотеки L4. Это позволяет выполнить собранный контейнерный filesystem без локальных вычислений, но не даёт Docker-изоляцию.

        Используйте только образ из собственного публичного GHCR package и только собственный голос либо голос с явным разрешением владельца. Не публикуйте временную Gradio-ссылку и завершайте runtime после работы.
        """
    ),
    markdown(
        """
        ## 1. Настройка

        Перед первым запуском:

        1. загрузите проект в GitHub и дождитесь зелёного workflow **Validate and publish GPU image**;
        2. сделайте GHCR package публичным, чтобы Colab мог скачать образ без токена;
        3. при необходимости укажите форк в `REPO_URL`; по умолчанию уже выбран опубликованный репозиторий проекта;
        4. выберите `Runtime -> Change runtime type -> L4 GPU`.

        Оставьте `CONTAINER_IMAGE_OVERRIDE` пустым: notebook сам выберет неизменяемый тег `sha-<commit>`. Укажите override только для осознанной проверки другого публичного образа.
        """
    ),
    code(
        """
        from pathlib import Path

        REPO_URL = "https://github.com/nahmax/voice-tts.git"  # @param {type:"string"}
        GIT_REF = "main"  # @param {type:"string"}
        EXECUTION_MODE = "udocker"  # @param ["udocker", "native"]
        CONTAINER_IMAGE_OVERRIDE = ""  # @param {type:"string"}
        REQUIRE_L4 = True  # @param {type:"boolean"}
        UDOCKER_VERSION = "1.3.17"
        UDOCKER_EXECMODE = "P1"
        CONTAINER_NAME = "voice-tts-colab"

        APP_DIR = Path("/content/voice-tts")
        COSYVOICE_REPO = Path("/content/CosyVoice")
        COLAB_PYTHON = Path("/content/voice-tts-conda/bin/python")
        UDOCKER_DIR = Path("/content/udocker")

        if EXECUTION_MODE not in {"udocker", "native"}:
            raise ValueError("EXECUTION_MODE must be 'udocker' or 'native'.")
        print("Execution mode:", EXECUTION_MODE)
        print("Repository:", REPO_URL)
        """
    ),
    markdown(
        """
        ## 2. Проверка удалённого GPU

        Ячейка намеренно отказывается работать вне hosted Google Colab. На вашем Windows-ПК она ничего не запускает.
        """
    ),
    code(
        """
        import subprocess

        try:
            import google.colab  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Этот notebook нужно запускать в hosted Google Colab.") from exc

        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        print(gpu.stdout.strip())
        if REQUIRE_L4 and "L4" not in gpu.stdout:
            raise RuntimeError("Выделен не L4. Смените runtime или поставьте REQUIRE_L4 = False.")
        """
    ),
    markdown("## 3. Получение точного Git-коммита и соответствующего GHCR-образа"),
    code(
        """
        import re
        import subprocess

        if not (APP_DIR / ".git").exists():
            subprocess.run(["git", "clone", "--filter=blob:none", REPO_URL, str(APP_DIR)], check=True)

        subprocess.run(["git", "-C", str(APP_DIR), "fetch", "origin", GIT_REF, "--depth", "1"], check=True)
        subprocess.run(["git", "-C", str(APP_DIR), "checkout", "--detach", "FETCH_HEAD"], check=True)
        commit = subprocess.run(
            ["git", "-C", str(APP_DIR), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        repo_match = re.fullmatch(r"https://github\\.com/([^/]+)/([^/]+?)(?:\\.git)?", REPO_URL.rstrip("/"))
        if not repo_match:
            raise ValueError("Для автоматического GHCR-тега нужен URL вида https://github.com/OWNER/REPO.git")
        repo_owner, repo_name = repo_match.groups()
        image_repository = f"ghcr.io/{repo_owner}/{repo_name}".lower()
        container_image = CONTAINER_IMAGE_OVERRIDE.strip() or f"{image_repository}:sha-{commit[:12]}"

        print("Checked out Git commit:", commit)
        print("Container image:", container_image)
        """
    ),
    markdown(
        """
        ## 4. Хранилище: Google Drive с локальным fallback

        По умолчанию notebook запрашивает доступ к `MyDrive/Voice TTS/`. Если Google DriveFS возвращает `mount failed`, работа продолжается во временном `/content/voice-tts-data`; такие данные исчезнут после удаления runtime.
        """
    ),
    code(
        """
        from google.colab import drive  # type: ignore

        USE_GOOGLE_DRIVE = True  # @param {type:"boolean"}
        STORAGE_IS_PERSISTENT = False
        if USE_GOOGLE_DRIVE:
            try:
                drive.mount("/content/drive")
                STORAGE_ROOT = Path("/content/drive/MyDrive/Voice TTS")
                STORAGE_IS_PERSISTENT = True
            except ValueError as exc:
                print(f"WARNING: Google Drive mount failed ({exc}); using temporary Colab storage.")
                STORAGE_ROOT = Path("/content/voice-tts-data")
        else:
            STORAGE_ROOT = Path("/content/voice-tts-data")

        DATA_DIR = STORAGE_ROOT
        MODEL_DIR = STORAGE_ROOT / "models" / "Fun-CosyVoice3-0.5B"
        for folder in [DATA_DIR / "voices", DATA_DIR / "runs", MODEL_DIR]:
            folder.mkdir(parents=True, exist_ok=True)
        print("Storage workspace:", STORAGE_ROOT)
        print("Persistent:", STORAGE_IS_PERSISTENT)
        """
    ),
    markdown(
        """
        ### 4a. Необязательный массовый импорт голосов

        Для обычной работы загружайте или записывайте голос в Gradio. Если у вас есть подготовленный `local_voice_library_upload.zip`, поставьте `IMPORT_LIBRARY_ZIP = True`. Архив безопасно проверяется перед копированием в Drive.
        """
    ),
    code(
        """
        import sys

        IMPORT_LIBRARY_ZIP = False  # @param {type:"boolean"}
        OVERWRITE_IMPORTED_VOICES = False  # @param {type:"boolean"}
        if IMPORT_LIBRARY_ZIP:
            from google.colab import files  # type: ignore

            uploaded = files.upload()
            if not uploaded:
                raise RuntimeError("ZIP не выбран.")
            zip_path = Path(next(iter(uploaded)))
            sys.path.insert(0, str(APP_DIR))
            from voice_tts.library import import_voice_library_zip

            result = import_voice_library_zip(zip_path, DATA_DIR, overwrite=OVERWRITE_IMPORTED_VOICES)
            zip_path.unlink(missing_ok=True)
            print("Imported:", result)
        else:
            print("ZIP import skipped; use Gradio for a single voice.")
        """
    ),
    markdown(
        """
        ## 5. Подготовка runtime и проверка CUDA

        В режиме `udocker` ячейка скачивает опубликованный образ, извлекает его во временный `/content/udocker`, подключает NVIDIA-библиотеки и проверяет CUDA **внутри образа**. Образ должен быть публичным и уже собранным GitHub Actions для напечатанного SHA-тега.

        В режиме `native` вместо этого создаётся временное Python 3.10 окружение. В обоих режимах веса модели загружаются на Drive один раз.
        """
    ),
    code(
        """
        import os
        import subprocess
        import sys

        MODEL_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
        MODEL_REVISION = "29e01c4e8d000f4bcd70751be16fa94bf3d85a18"
        common_container_env = {
            "COSYVOICE_REPO": "/opt/CosyVoice",
            "MODEL_ID": MODEL_ID,
            "MODEL_REVISION": MODEL_REVISION,
            "MODEL_DIR": "/models/Fun-CosyVoice3-0.5B",
            "DATA_DIR": "/data",
            "HF_HOME": "/models/huggingface",
            "WHISPER_CACHE_DIR": "/models/whisper",
            "REQUIRE_CUDA": "1",
            "COSYVOICE_FP16": "1",
            "WHISPER_MODEL": "base",
        }

        def make_udocker_run(command, extra_env=None):
            merged_env = {**common_container_env, **(extra_env or {})}
            return [
                "udocker", "--allow-root", "run",
                f"--volume={STORAGE_ROOT}:/data",
                f"--volume={STORAGE_ROOT / 'models'}:/models",
                "--workdir=/workspace",
                *[f"--env={key}={value}" for key, value in merged_env.items()],
                CONTAINER_NAME,
                *command,
            ]

        if EXECUTION_MODE == "udocker":
            subprocess.run([sys.executable, "-m", "pip", "install", f"udocker=={UDOCKER_VERSION}"], check=True)
            udocker_env = os.environ.copy()
            udocker_env["UDOCKER_DIR"] = str(UDOCKER_DIR)
            subprocess.run(["udocker", "install"], env=udocker_env, check=True)

            udocker = ["udocker", "--allow-root"]
            subprocess.run([*udocker, "rm", "-f", CONTAINER_NAME], env=udocker_env, check=False)
            try:
                subprocess.run(
                    [*udocker, "pull", "--platform=linux/amd64", container_image],
                    env=udocker_env,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    "Не удалось скачать точный GHCR-образ. Дождитесь завершения GitHub Actions "
                    "и сделайте package публичным, затем повторите ячейку."
                ) from exc
            subprocess.run(
                [*udocker, "create", f"--name={CONTAINER_NAME}", container_image],
                env=udocker_env,
                check=True,
            )
            subprocess.run(
                [*udocker, "setup", f"--execmode={UDOCKER_EXECMODE}", "--nvidia", "--force", CONTAINER_NAME],
                env=udocker_env,
                check=True,
            )
            subprocess.run(
                make_udocker_run([
                    "python", "-c",
                    "from voice_tts.runtime import require_cuda, ensure_model_downloaded; "
                    "require_cuda(); print(ensure_model_downloaded())",
                ]),
                env=udocker_env,
                check=True,
            )
            print("OCI image and CUDA preflight passed:", container_image)
        elif EXECUTION_MODE == "native":
            native_env = os.environ.copy()
            native_env["COSYVOICE_REPO"] = str(COSYVOICE_REPO)
            subprocess.run(
                ["bash", str(APP_DIR / "scripts" / "bootstrap_colab.sh")],
                cwd=APP_DIR,
                env=native_env,
                check=True,
            )
            if not COLAB_PYTHON.exists():
                raise FileNotFoundError(COLAB_PYTHON)
            native_env.update({
                "MODEL_DIR": str(MODEL_DIR),
                "MODEL_ID": MODEL_ID,
                "MODEL_REVISION": MODEL_REVISION,
                "HF_HOME": str(STORAGE_ROOT / "models" / "huggingface"),
                "REQUIRE_CUDA": "1",
            })
            subprocess.run(
                [
                    str(COLAB_PYTHON), "-c",
                    "from voice_tts.runtime import require_cuda, ensure_model_downloaded; "
                    "require_cuda(); print(ensure_model_downloaded())",
                ],
                cwd=APP_DIR,
                env=native_env,
                check=True,
            )
            print("Native CUDA preflight passed.")
        """
    ),
    markdown(
        """
        ## 6. Запуск веб-интерфейса

        Ячейка запускает Gradio в фоне и печатает временную ссылку, логин и случайный пароль. Первый старт может занять несколько минут. Повторный запуск ячейки сначала завершает предыдущий процесс.
        """
    ),
    code(
        """
        import os
        import re
        import secrets
        import signal
        import subprocess
        import time

        pid_file = Path("/content/voice_tts_app.pid")
        log_file = Path("/content/voice_tts_app.log")

        def stop_previous_app():
            if not pid_file.exists():
                return
            old_pid = int(pid_file.read_text().strip())
            try:
                os.killpg(old_pid, signal.SIGTERM)
                time.sleep(2)
            except ProcessLookupError:
                pass
            pid_file.unlink(missing_ok=True)

        stop_previous_app()
        gradio_username = "voice"
        gradio_password = secrets.token_urlsafe(18)

        if EXECUTION_MODE == "udocker":
            process_env = udocker_env.copy()
            command = make_udocker_run(
                ["python", "app.py", "--share", "--port", "7860"],
                {
                    "GRADIO_USERNAME": gradio_username,
                    "GRADIO_PASSWORD": gradio_password,
                    "GRADIO_SHARE": "1",
                },
            )
            process_cwd = APP_DIR
        elif EXECUTION_MODE == "native":
            process_env = os.environ.copy()
            process_env.update({
                "COSYVOICE_REPO": str(COSYVOICE_REPO),
                "MODEL_DIR": str(MODEL_DIR),
                "MODEL_REVISION": MODEL_REVISION,
                "DATA_DIR": str(DATA_DIR),
                "HF_HOME": str(STORAGE_ROOT / "models" / "huggingface"),
                "WHISPER_CACHE_DIR": str(STORAGE_ROOT / "models" / "whisper"),
                "REQUIRE_CUDA": "1",
                "COSYVOICE_FP16": "1",
                "WHISPER_MODEL": "base",
                "GRADIO_USERNAME": gradio_username,
                "GRADIO_PASSWORD": gradio_password,
                "GRADIO_SHARE": "1",
            })
            command = [str(COLAB_PYTHON), "app.py", "--share", "--port", "7860"]
            process_cwd = APP_DIR

        with log_file.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=process_cwd,
                env=process_env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pid_file.write_text(str(process.pid), encoding="utf-8")

        public_url = None
        for _ in range(180):
            time.sleep(2)
            log_text = log_file.read_text(encoding="utf-8", errors="replace") if log_file.exists() else ""
            match = re.search(r"https://[a-zA-Z0-9.-]+\\.gradio\\.live", log_text)
            if match:
                public_url = match.group(0)
                break
            if process.poll() is not None:
                raise RuntimeError("Gradio завершился. Последние строки лога:\\n" + log_text[-6000:])
        if not public_url:
            raise TimeoutError("Ссылка Gradio не появилась. Проверьте /content/voice_tts_app.log")

        print("Откройте интерфейс:", public_url)
        print("Логин:", gradio_username)
        print("Пароль:", gradio_password)
        print("Режим:", EXECUTION_MODE)
        print("PID:", process.pid)
        """
    ),
    markdown(
        """
        ## Как пользоваться UI

        1. Поставьте галочку согласия.
        2. Загрузите аудио или запишите 3–30 секунд чистой речи с микрофона.
        3. При желании нажмите **Распознать референс**; иначе Whisper сделает это при генерации.
        4. Введите текст и нажмите **Сгенерировать речь**.
        5. Прослушайте WAV. Результат появится в `MyDrive/Voice TTS/runs/`, а сохранённый голос — в `MyDrive/Voice TTS/voices/`.
        """
    ),
    markdown("## 7. Проверка сохранённых результатов"),
    code(
        """
        import wave

        run_files = sorted(
            (DATA_DIR / "runs").glob("*/output.wav"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        print("Generated WAV files:", len(run_files))
        for path in run_files[:10]:
            with wave.open(str(path), "rb") as wav:
                duration = wav.getnframes() / wav.getframerate()
            print(f"{path} | {duration:.2f} s")
        if not run_files:
            print("Сначала выполните одну генерацию через UI.")
        """
    ),
    markdown(
        """
        ## 8. Остановка

        `Run all` намеренно оставляет приложение запущенным. После работы поставьте `STOP_APP = True`, выполните эту ячейку отдельно, а затем выберите `Runtime -> Disconnect and delete runtime`, чтобы освободить L4.
        """
    ),
    code(
        """
        import os
        import signal

        STOP_APP = False  # @param {type:"boolean"}
        pid_file = Path("/content/voice_tts_app.pid")
        if STOP_APP and pid_file.exists():
            pid = int(pid_file.read_text().strip())
            try:
                os.killpg(pid, signal.SIGTERM)
                print("Stopped process group", pid)
            except ProcessLookupError:
                print("Process already stopped")
            pid_file.unlink(missing_ok=True)
        elif STOP_APP:
            print("No running app PID found")
        else:
            print("App left running. Set STOP_APP = True and run this cell when finished.")
        """
    ),
    markdown(
        """
        ## Критерий сквозной проверки

        Запуск считается успешным только если одновременно выполнены четыре условия:

        - preflight напечатал модель L4 и подтвердил CUDA внутри выбранного runtime;
        - Gradio открылся по временной ссылке с паролем;
        - UI воспроизвёл сгенерированный WAV;
        - тот же WAV появился в `MyDrive/Voice TTS/runs/`.

        Если возникает ошибка, сохраните последние 100 строк `/content/voice_tts_app.log` и полный вывод ячейки подготовки runtime.
        """
    ),
]


NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "name": "voice_tts_colab_gpu.ipynb",
            "provenance": [],
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(NOTEBOOK, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("Wrote", OUTPUT)


if __name__ == "__main__":
    main()
