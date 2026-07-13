# Local Voice Library

Это локальная папка на компьютере для подготовки голосов перед загрузкой в Google Drive.

Главная папка:

    C:\disk_d\arbeit\DS\prog\task\voice-tts\local_voice_library

Куда класть записи:

    local_voice_library\voices\voice_0\
    local_voice_library\voices\voice_1\
    local_voice_library\voices\voice_2\
    local_voice_library\voices\voice_3\
    local_voice_library\voices\voice_4\
    local_voice_library\voices\voice_5\

Можно создавать свои папки внутри voices, например:

    local_voice_library\voices\max_main\
    local_voice_library\voices\max_calm\
    local_voice_library\voices\narrator_fast\

Внутри каждой папки может быть один или несколько файлов записи голоса.

Поддерживаемые форматы:
- wav
- mp3
- m4a
- flac
- ogg

Как загрузить на Google Drive:

1. Разложи файлы по подпапкам внутри local_voice_library\voices.
2. Запусти make_voice_library_zip.ps1 из корня проекта.
3. Получится файл local_voice_library_upload.zip.
4. Открой output/jupyter-notebook/voice_tts_colab_gpu.ipynb в Google Colab.
5. В ячейке `4a. Необязательный массовый импорт` включи IMPORT_LIBRARY_ZIP = True.
6. Загрузи local_voice_library_upload.zip через upload dialog.
7. Notebook безопасно импортирует голоса в MyDrive/Voice TTS/voices/.

Основной notebook `output/jupyter-notebook/voice_tts_colab_gpu.ipynb` и Gradio UI уже умеют сохранять новые записи прямо в `MyDrive/Voice TTS/voices/`, поэтому ZIP-путь необязателен. Он остаётся полезным для массового переноса заранее подготовленных папок через ячейку 4a.
