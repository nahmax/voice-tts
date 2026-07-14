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
        ENABLE_VOXCPM = True  # @param {type:"boolean"}
        PRELOAD_COSYVOICE = False  # @param {type:"boolean"}
        CONTAINER_IMAGE_OVERRIDE = ""  # @param {type:"string"}
        REQUIRE_L4 = True  # @param {type:"boolean"}
        UDOCKER_VERSION = "1.3.17"
        UDOCKER_EXECMODE = "P1"
        CONTAINER_NAME = "voice-tts-colab"
        PROGRESS_HEARTBEAT_SECONDS = 15
        GRADIO_START_TIMEOUT_SECONDS = 600

        APP_DIR = Path("/content/voice-tts")
        COSYVOICE_REPO = Path("/content/CosyVoice")
        COLAB_PYTHON = Path("/content/voice-tts-conda/bin/python")
        UDOCKER_DIR = Path("/content/udocker")
        OCI_LAYOUT_DIR = Path("/content/voice-tts-oci-layout")
        OCI_ARCHIVE = Path("/content/voice-tts-image.oci.tar")
        UDOCKER_READY_MARKER = Path("/content/voice-tts-udocker-ready.txt")
        NATIVE_READY_MARKER = Path("/content/voice-tts-native-ready.txt")
        VOXCPM_VERSION = "2.0.3"
        VOXCPM_MODEL_ID = "openbmb/VoxCPM2"
        VOXCPM_ENV = Path(f"/content/voice-tts-voxcpm-{VOXCPM_VERSION}")
        VOXCPM_PYTHON = VOXCPM_ENV / "bin" / "python"
        VOXCPM_READY_MARKER = VOXCPM_ENV / ".voice_tts_ready"
        VOXCPM_WORKER_PORT = 8765
        VOXCPM_WORKER_URL = f"http://127.0.0.1:{VOXCPM_WORKER_PORT}"
        VOXCPM_WORKER_PID_FILE = Path("/content/voice_tts_voxcpm.pid")
        VOXCPM_WORKER_LOG = Path("/content/voice_tts_voxcpm.log")

        if EXECUTION_MODE not in {"udocker", "native"}:
            raise ValueError("EXECUTION_MODE must be 'udocker' or 'native'.")
        print("Execution mode:", EXECUTION_MODE)
        print("VoxCPM2 enabled:", ENABLE_VOXCPM)
        print("CosyVoice preload:", PRELOAD_COSYVOICE)
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
        VOXCPM_HF_HOME = STORAGE_ROOT / "models" / "voxcpm-huggingface"
        for folder in [DATA_DIR / "voices", DATA_DIR / "runs", MODEL_DIR, VOXCPM_HF_HOME]:
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

        В режиме `udocker` ячейка скачивает опубликованный образ с возобновлением каждого OCI-слоя после сетевого обрыва, загружает его в `/content/udocker`, подключает NVIDIA-библиотеки и проверяет CUDA **внутри образа**. Размер и SHA-256 каждого слоя проверяются до импорта. Образ должен быть публичным и уже собранным GitHub Actions для напечатанного SHA-тега.

        Ячейка показывает номер текущего этапа, сколько этапов осталось и каждые 15 секунд подтверждает, что дочерний процесс жив. Во время скачивания OCI-образа дополнительно печатаются общий процент, объём, скорость и расчётный ETA. Повторный запуск в том же Colab runtime использует уже подготовленный контейнер или native-окружение.

        В режиме `native` вместо контейнера создаётся временное Python 3.10 окружение. В обоих режимах веса модели загружаются на Drive один раз. ETA является оценкой по текущей скорости сети и может меняться.

        VoxCPM2 устанавливается в отдельное окружение, потому что ему нужны PyTorch >=2.5 и Gradio 6, тогда как закреплённый CosyVoice использует PyTorch 2.3.1 и Gradio 5.4. Отдельный localhost-worker сохраняет совместимость обоих движков; при переключении интерфейс выгружает неиспользуемую TTS-модель из GPU.
        """
    ),
    code(
        """
        import os
        import json
        import secrets
        import signal
        import subprocess
        import sys
        import time
        import urllib.error
        import urllib.request

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
            "WHISPER_MODEL": "medium",
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

        def format_elapsed(seconds):
            total_seconds = max(0, int(seconds))
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours:
                return f"{hours:d} ч {minutes:02d} мин {seconds:02d} с"
            if minutes:
                return f"{minutes:d} мин {seconds:02d} с"
            return f"{seconds:d} с"

        def run_with_heartbeat(command, label, *, cwd=None, env=None, check=True):
            started = time.monotonic()
            process = subprocess.Popen(command, cwd=cwd, env=env)
            print(f"    PID {process.pid}; вывод процесса остаётся видимым ниже.", flush=True)
            try:
                while True:
                    try:
                        return_code = process.wait(timeout=PROGRESS_HEARTBEAT_SECONDS)
                        break
                    except subprocess.TimeoutExpired:
                        elapsed = format_elapsed(time.monotonic() - started)
                        print(
                            f"    … всё ещё работает: {label} | прошло {elapsed} | PID {process.pid} активен",
                            flush=True,
                        )
            except KeyboardInterrupt:
                process.terminate()
                raise
            elapsed = format_elapsed(time.monotonic() - started)
            if return_code != 0:
                print(f"    ✗ ошибка на этапе: {label} | прошло {elapsed} | код {return_code}", flush=True)
                if check:
                    raise subprocess.CalledProcessError(return_code, command)
            else:
                print(f"    ✓ завершено: {label} | {elapsed}", flush=True)
            return return_code

        class StageProgress:
            def __init__(self, total):
                self.total = total
                self.current = 0
                self.started = time.monotonic()

            def begin(self, label):
                self.current += 1
                remaining = self.total - self.current
                print(
                    f"\\n[{self.current}/{self.total}] {label} | после этого останется этапов: {remaining}",
                    flush=True,
                )

            def run(self, label, command, *, cwd=None, env=None, check=True):
                self.begin(label)
                return run_with_heartbeat(command, label, cwd=cwd, env=env, check=check)

            def finish(self):
                elapsed = format_elapsed(time.monotonic() - self.started)
                print(f"\\nГотово: выполнено {self.total}/{self.total} этапов за {elapsed}.", flush=True)

        if EXECUTION_MODE == "udocker":
            udocker_env = os.environ.copy()
            udocker_env["UDOCKER_DIR"] = str(UDOCKER_DIR)
            udocker = ["udocker", "--allow-root"]
            image_is_ready = (
                UDOCKER_READY_MARKER.is_file()
                and UDOCKER_READY_MARKER.read_text(encoding="utf-8").strip() == container_image
            )
            progress = StageProgress(5 if image_is_ready else 9)
            progress.run(
                "Установка команды udocker",
                [
                    sys.executable,
                    "-m", "pip", "install", "--disable-pip-version-check",
                    f"udocker=={UDOCKER_VERSION}",
                ],
            )
            progress.run(
                "Инициализация udocker",
                ["udocker", "--allow-root", "install"],
                env=udocker_env,
            )

            if image_is_ready:
                print(
                    f"\\nКонтейнер {container_image} уже подготовлен в этом runtime; повторный импорт пропущен.",
                    flush=True,
                )
            else:
                subprocess.run([*udocker, "rm", "-f", CONTAINER_NAME], env=udocker_env, check=False)
                subprocess.run([*udocker, "rmi", container_image], env=udocker_env, check=False)
                progress.run(
                    "Скачивание и проверка OCI-образа",
                    [
                        sys.executable,
                        str(APP_DIR / "scripts" / "pull_oci_resumable.py"),
                        container_image,
                        "--layout-dir", str(OCI_LAYOUT_DIR),
                        "--output", str(OCI_ARCHIVE),
                    ],
                )
                progress.run(
                    "Импорт OCI-образа в udocker",
                    [*udocker, "load", "-i", str(OCI_ARCHIVE), image_repository],
                    env=udocker_env,
                )
                OCI_ARCHIVE.unlink(missing_ok=True)
                progress.run(
                    "Создание контейнера",
                    [*udocker, "create", f"--name={CONTAINER_NAME}", container_image],
                    env=udocker_env,
                )
                progress.run(
                    "Подключение NVIDIA-библиотек",
                    [*udocker, "setup", f"--execmode={UDOCKER_EXECMODE}", "--nvidia", "--force", CONTAINER_NAME],
                    env=udocker_env,
                )
            progress.run(
                "Проверка CUDA внутри контейнера",
                make_udocker_run([
                    "python", "-c",
                    "from voice_tts.runtime import require_cuda, gpu_summary; "
                    "require_cuda(); print('GPU:', gpu_summary())",
                ]),
                env=udocker_env,
            )
            UDOCKER_READY_MARKER.write_text(container_image + "\\n", encoding="utf-8")
            progress.run(
                "Загрузка или проверка файлов модели",
                (make_udocker_run([
                    "python", "-c",
                    "from voice_tts.runtime import ensure_model_downloaded; "
                    "print('Model files:', ensure_model_downloaded())",
                ]) if PRELOAD_COSYVOICE else [
                    sys.executable, "-c",
                    "print('CosyVoice 3 fallback будет загружен лениво при первом выборе.')",
                ]),
                env=udocker_env,
            )
            progress.run(
                "Загрузка модели в память GPU",
                (make_udocker_run([
                    "python", "-c",
                    "from voice_tts.runtime import require_cuda, get_model; "
                    "require_cuda(); print('Model:', type(get_model()).__name__)",
                ]) if PRELOAD_COSYVOICE else [
                    sys.executable, "-c",
                    "print('Предзагрузка CosyVoice 3 в GPU пропущена.')",
                ]),
                env=udocker_env,
            )
            progress.finish()
            print("OCI image and CUDA preflight passed:", container_image)
        elif EXECUTION_MODE == "native":
            native_env = os.environ.copy()
            native_env["COSYVOICE_REPO"] = str(COSYVOICE_REPO)
            native_is_ready = (
                NATIVE_READY_MARKER.is_file()
                and NATIVE_READY_MARKER.read_text(encoding="utf-8").strip() == commit
                and COLAB_PYTHON.exists()
            )
            progress = StageProgress(3 if native_is_ready else 4)
            if native_is_ready:
                print(
                    f"Native-окружение для {commit[:12]} уже готово в этом runtime; установка пропущена.",
                    flush=True,
                )
            else:
                progress.run(
                    "Установка native Python/CosyVoice runtime",
                    ["bash", str(APP_DIR / "scripts" / "bootstrap_colab.sh")],
                    cwd=APP_DIR,
                    env=native_env,
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
            progress.run(
                "Проверка CUDA в native-окружении",
                [
                    str(COLAB_PYTHON), "-c",
                    "from voice_tts.runtime import require_cuda, gpu_summary; "
                    "require_cuda(); print('GPU:', gpu_summary())",
                ],
                cwd=APP_DIR,
                env=native_env,
            )
            NATIVE_READY_MARKER.write_text(commit + "\\n", encoding="utf-8")
            progress.run(
                "Загрузка или проверка файлов модели",
                ([
                    str(COLAB_PYTHON), "-c",
                    "from voice_tts.runtime import ensure_model_downloaded; "
                    "print('Model files:', ensure_model_downloaded())",
                ] if PRELOAD_COSYVOICE else [
                    sys.executable, "-c",
                    "print('CosyVoice 3 fallback будет загружен лениво при первом выборе.')",
                ]),
                cwd=APP_DIR,
                env=native_env,
            )
            progress.run(
                "Загрузка модели в память GPU",
                ([
                    str(COLAB_PYTHON), "-c",
                    "from voice_tts.runtime import require_cuda, get_model; "
                    "require_cuda(); print('Model:', type(get_model()).__name__)",
                ] if PRELOAD_COSYVOICE else [
                    sys.executable, "-c",
                    "print('Предзагрузка CosyVoice 3 в GPU пропущена.')",
                ]),
                cwd=APP_DIR,
                env=native_env,
            )
            progress.finish()
            print("Native CUDA preflight passed.")

        voxcpm_worker_token = ""
        if ENABLE_VOXCPM:
            voxcpm_is_ready = (
                VOXCPM_PYTHON.is_file()
                and VOXCPM_READY_MARKER.is_file()
                and VOXCPM_READY_MARKER.read_text(encoding="utf-8").strip() == VOXCPM_VERSION
            )
            voxcpm_progress = StageProgress(2 if voxcpm_is_ready else 4)
            if voxcpm_is_ready:
                print(
                    f"\\nVoxCPM2 {VOXCPM_VERSION} уже установлен в этом runtime; установка пропущена.",
                    flush=True,
                )
            else:
                voxcpm_progress.run(
                    "Создание отдельного VoxCPM2 Python-окружения",
                    [sys.executable, "-m", "venv", "--system-site-packages", str(VOXCPM_ENV)],
                )
                voxcpm_progress.run(
                    f"Установка VoxCPM2 {VOXCPM_VERSION}",
                    [
                        str(VOXCPM_PYTHON), "-m", "pip", "install",
                        "--disable-pip-version-check", "--upgrade", f"voxcpm=={VOXCPM_VERSION}",
                    ],
                )

            voxcpm_progress.run(
                "Проверка отдельного VoxCPM2 CUDA-runtime",
                [
                    str(VOXCPM_PYTHON), "-c",
                    "import torch, voxcpm; "
                    "assert torch.cuda.is_available(), 'CUDA unavailable'; "
                    "print('VoxCPM package:', voxcpm.__file__); "
                    "print('torch:', torch.__version__); "
                    "print('gpu:', torch.cuda.get_device_name(0))",
                ],
            )
            VOXCPM_READY_MARKER.write_text(VOXCPM_VERSION + "\\n", encoding="utf-8")

            if VOXCPM_WORKER_PID_FILE.exists():
                old_worker_pid = int(VOXCPM_WORKER_PID_FILE.read_text().strip())
                try:
                    os.killpg(old_worker_pid, signal.SIGTERM)
                    time.sleep(2)
                except ProcessLookupError:
                    pass
                VOXCPM_WORKER_PID_FILE.unlink(missing_ok=True)

            voxcpm_progress.begin("Запуск, загрузка весов и прогрев VoxCPM2")
            voxcpm_worker_token = secrets.token_urlsafe(24)
            voxcpm_env = os.environ.copy()
            voxcpm_env.update({
                "PYTHONUNBUFFERED": "1",
                "HF_HOME": str(VOXCPM_HF_HOME),
                "VOXCPM_MODEL_ID": VOXCPM_MODEL_ID,
                "VOXCPM_WORKER_TOKEN": voxcpm_worker_token,
            })
            with VOXCPM_WORKER_LOG.open("w", encoding="utf-8") as worker_log:
                voxcpm_process = subprocess.Popen(
                    [
                        str(VOXCPM_PYTHON), str(APP_DIR / "scripts" / "voxcpm_worker.py"),
                        "--host", "127.0.0.1", "--port", str(VOXCPM_WORKER_PORT), "--preload",
                    ],
                    cwd=APP_DIR,
                    env=voxcpm_env,
                    stdout=worker_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            VOXCPM_WORKER_PID_FILE.write_text(str(voxcpm_process.pid), encoding="utf-8")

            worker_started = time.monotonic()
            worker_timeout_seconds = 1800
            next_heartbeat = worker_started
            printed_log_chars = 0
            worker_state = "starting"
            while True:
                if VOXCPM_WORKER_LOG.exists():
                    worker_log_text = VOXCPM_WORKER_LOG.read_text(encoding="utf-8", errors="replace")
                    if len(worker_log_text) > printed_log_chars:
                        print(worker_log_text[printed_log_chars:], end="", flush=True)
                        printed_log_chars = len(worker_log_text)
                if voxcpm_process.poll() is not None:
                    raise RuntimeError(
                        "VoxCPM2 worker завершился. Последний лог:\\n" +
                        (worker_log_text[-6000:] if 'worker_log_text' in locals() else "")
                    )
                try:
                    request = urllib.request.Request(
                        f"{VOXCPM_WORKER_URL}/health",
                        headers={"X-Voice-TTS-Token": voxcpm_worker_token},
                    )
                    with urllib.request.urlopen(request, timeout=3) as response:
                        worker_health = json.loads(response.read().decode("utf-8"))
                    worker_state = str(worker_health.get("status", "unknown"))
                    if worker_state == "ready":
                        break
                    if worker_state == "error":
                        raise RuntimeError(f"VoxCPM2 не загрузился: {worker_health.get('error')}")
                except (urllib.error.URLError, TimeoutError):
                    worker_state = "starting"

                now = time.monotonic()
                elapsed = now - worker_started
                if elapsed >= worker_timeout_seconds:
                    raise TimeoutError("VoxCPM2 не загрузился за 30 минут. Проверьте /content/voice_tts_voxcpm.log")
                if now >= next_heartbeat:
                    remaining = worker_timeout_seconds - elapsed
                    print(
                        f"    … VoxCPM2: {worker_state} | прошло {format_elapsed(elapsed)} | "
                        f"до таймаута {format_elapsed(remaining)} | PID {voxcpm_process.pid} активен",
                        flush=True,
                    )
                    next_heartbeat = now + PROGRESS_HEARTBEAT_SECONDS
                time.sleep(2)

            print(
                f"    ✓ VoxCPM2 готов за {format_elapsed(time.monotonic() - worker_started)}; "
                f"worker PID {voxcpm_process.pid}",
                flush=True,
            )
            voxcpm_progress.finish()
        else:
            print("VoxCPM2 отключён; в UI будет доступен только CosyVoice 3.")
        """
    ),
    markdown(
        """
        ## 6. Запуск веб-интерфейса

        Ячейка запускает Gradio в фоне и печатает временную ссылку, логин и случайный пароль. Пока ссылка создаётся, каждые 10 секунд выводятся прошедшее время, оставшееся время до таймаута и подтверждение, что PID приложения активен. Повторный запуск ячейки сначала завершает предыдущий процесс.
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
        optional_voxcpm_env = {}
        if ENABLE_VOXCPM:
            optional_voxcpm_env = {
                "VOXCPM_WORKER_URL": VOXCPM_WORKER_URL,
                "VOXCPM_WORKER_TOKEN": voxcpm_worker_token,
                "DEFAULT_TTS_ENGINE": "voxcpm2",
            }

        if EXECUTION_MODE == "udocker":
            process_env = udocker_env.copy()
            command = make_udocker_run(
                ["python", "app.py", "--host", "127.0.0.1", "--share", "--port", "7860"],
                {
                    "GRADIO_USERNAME": gradio_username,
                    "GRADIO_PASSWORD": gradio_password,
                    "GRADIO_SHARE": "1",
                    **optional_voxcpm_env,
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
                "WHISPER_MODEL": "medium",
                "GRADIO_USERNAME": gradio_username,
                "GRADIO_PASSWORD": gradio_password,
                "GRADIO_SHARE": "1",
                **optional_voxcpm_env,
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
        poll_interval = 2
        max_attempts = max(1, GRADIO_START_TIMEOUT_SECONDS // poll_interval)
        startup_started = time.monotonic()
        print(
            f"Gradio запускается: PID {process.pid}; таймаут {format_elapsed(GRADIO_START_TIMEOUT_SECONDS)}.",
            flush=True,
        )
        for attempt in range(1, max_attempts + 1):
            time.sleep(poll_interval)
            log_text = log_file.read_text(encoding="utf-8", errors="replace") if log_file.exists() else ""
            match = re.search(r"https://[a-zA-Z0-9.-]+\\.gradio\\.live", log_text)
            if match:
                public_url = match.group(0)
                break
            if process.poll() is not None:
                raise RuntimeError("Gradio завершился. Последние строки лога:\\n" + log_text[-6000:])
            if attempt % max(1, 10 // poll_interval) == 0:
                elapsed_seconds = time.monotonic() - startup_started
                remaining_seconds = max(0, GRADIO_START_TIMEOUT_SECONDS - elapsed_seconds)
                print(
                    f"… Gradio всё ещё запускается | прошло {format_elapsed(elapsed_seconds)} | "
                    f"до таймаута {format_elapsed(remaining_seconds)} | PID {process.pid} активен",
                    flush=True,
                )
        if not public_url:
            raise TimeoutError(
                "Ссылка Gradio не появилась за "
                f"{format_elapsed(GRADIO_START_TIMEOUT_SECONDS)}. "
                "Проверьте /content/voice_tts_app.log"
            )

        print("Gradio готов за", format_elapsed(time.monotonic() - startup_started))
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
        2. Оставьте **VoxCPM2 2B** для основного 48-кГц режима либо выберите **CosyVoice 3** как fallback.
        3. Загрузите аудио любой длины и выберите 8–15 секунд чистой речи (технический диапазон референса: 3–30 секунд).
        4. При желании нажмите **Распознать референс**; иначе Whisper сделает это при генерации.
        5. Введите текст и нажмите **Сгенерировать речь**.
        6. Прослушайте WAV. Результат и выбранный движок записываются в `MyDrive/Voice TTS/runs/`, а сохранённый голос — в `MyDrive/Voice TTS/voices/`.
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
        if STOP_APP:
            stopped_any = False
            for pid_path, label in [
                (Path("/content/voice_tts_app.pid"), "Gradio"),
                (Path("/content/voice_tts_voxcpm.pid"), "VoxCPM2 worker"),
            ]:
                if not pid_path.exists():
                    continue
                pid = int(pid_path.read_text().strip())
                try:
                    os.killpg(pid, signal.SIGTERM)
                    print(f"Stopped {label} process group", pid)
                except ProcessLookupError:
                    print(f"{label} was already stopped")
                pid_path.unlink(missing_ok=True)
                stopped_any = True
            if not stopped_any:
                print("No running app or VoxCPM2 worker PID found")
        else:
            print("App and VoxCPM2 worker left running. Set STOP_APP = True when finished.")
        """
    ),
    markdown(
        """
        ## Критерий сквозной проверки

        Запуск считается успешным только если одновременно выполнены пять условий:

        - preflight напечатал модель L4 и подтвердил CUDA внутри выбранного runtime;
        - VoxCPM2 worker перешёл в состояние `ready` (если `ENABLE_VOXCPM=True`);
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
