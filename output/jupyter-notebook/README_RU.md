# Colab Voice TTS

Для нового workflow нужен только один notebook:

`voice_tts_colab_gpu.ipynb`

Он выполняет весь цикл:

1. Проверяет hosted Google Colab и L4.
2. Получает выбранный Git-коммит и вычисляет его точный GHCR-тег `sha-<12 символов>`.
3. Подключает Google Drive.
4. В основном режиме скачивает публичный OCI/Docker-образ из GHCR, запускает его через `udocker` и подключает NVIDIA-библиотеки L4.
5. Проверяет CUDA внутри образа; в резервном режиме создаёт изолированное Python 3.10 окружение.
6. Хранит модель в `MyDrive/Voice TTS/models/`.
7. Запускает Gradio для загрузки/записи голоса и ввода текста.
8. Сохраняет библиотеку голосов в `MyDrive/Voice TTS/voices/`.
9. Сохраняет результаты в `MyDrive/Voice TTS/runs/`.
10. Останавливает приложение и закрывает временную ссылку.

Hosted Colab не запускает Docker daemon. `udocker` исполняет доверенный контейнерный образ в user space без полноценной Docker-изоляции. Перед запуском дождитесь GitHub Actions, сделайте GHCR package публичным, задайте `REPO_URL`, оставьте `EXECUTION_MODE = "udocker"` и выполните notebook сверху вниз. Если этот runtime несовместим с `udocker`, смените режим на `native`.

Предыдущие раздельные setup/workflow notebooks удалены: весь актуальный запуск, включая необязательный импорт `local_voice_library_upload.zip`, находится в одном файле.

Полная инструкция, GitHub/GHCR и Docker-схема описаны в корневом `README.md`.
