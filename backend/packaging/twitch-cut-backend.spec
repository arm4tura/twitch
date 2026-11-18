# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller: бэкенд twitch-cut одним каталогом с исполняемым файлом.

ЗАЧЕМ. Раньше установщик при первом запуске качал uv, CPython 3.12, создавал
venv и ставил torch+CUDA (~10 ГБ, 10–15 минут). Теперь бэкенд собирается
заранее, в CI, и едет внутри установщика: Python и все зависимости уже
внутри, при первом запуске докачиваются только ONNX-модели (~430 МБ).

Сборка — `npm run build:backend` в desktop/ (scripts/build-backend.js), или руками
из backend/, в окружении из requirements-win.txt / requirements-mac.txt плюс
`pip install -e . --no-deps` и pyinstaller:
    pyinstaller --noconfirm --clean --distpath dist/<arch> packaging/twitch-cut-backend.spec
где <arch> — x64 или arm64. Результат: dist/<arch>/twitch-cut-backend/
(исполняемый файл + _internal/). electron-builder кладёт этот каталог в
resources/backend (см. desktop/package.json). Кросс-сборки нет: бэкенд под
каждую архитектуру собирается на машине этой архитектуры.

onedir, а не onefile: onefile распаковывает себя во временный каталог на
КАЖДОМ запуске — это секунды старта и сотни МБ во %TEMP%, а антивирусы
такие exe не любят.
"""

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

BACKEND = Path(SPECPATH).resolve().parent  # noqa: F821 — SPECPATH задаёт PyInstaller

# onnxruntime: libonnxruntime и провайдеры. В onnxruntime-directml модуль тот
# же, `onnxruntime`, и DirectML.dll лежит рядом в capi/ — забираем всё.
binaries = collect_dynamic_libs("onnxruntime")

datas = []
# Словари pymorphy3: pymorphy3 ищет их через entry points пакета словаря,
# поэтому нужны и сами файлы, и метаданные дистрибутива.
datas += collect_data_files("pymorphy3")
datas += collect_data_files("pymorphy3_dicts_ru")
datas += copy_metadata("pymorphy3_dicts_ru")
datas += copy_metadata("pymorphy3")
# Бандл-ffmpeg (imageio_ffmpeg/binaries/ffmpeg-*) — им режем, клеим и
# извлекаем аудио.
datas += collect_data_files("imageio_ffmpeg")
# Корневые сертификаты для HTTPS (скачивание моделей и TwitchDownloaderCLI):
# собранный Python на macOS системных сертификатов не видит.
datas += collect_data_files("certifi")
# Встроенный словарь мата (config.default_banwords_path ищет в sys._MEIPASS).
for name in ("banwords.txt", "banwords.example.txt"):
    if (BACKEND / name).exists():
        datas.append((str(BACKEND / name), "."))

hiddenimports = []
# Модули проекта импортируются лениво (движки, сервер, LLM) — статический
# анализ PyInstaller их не видит.
hiddenimports += collect_submodules("twitch_cut")
# uvicorn выбирает реализацию цикла/протоколов по строке во время запуска.
hiddenimports += collect_submodules("uvicorn")
hiddenimports += collect_submodules("websockets")
hiddenimports += ["certifi"]

excludes = [
    # Тяжёлые пакеты, которых в рантайме быть не должно. Если они случайно
    # стоят в сборочном окружении — не тащим их в бандл.
    "torch",
    "torchaudio",
    "whisperx",
    "faster_whisper",
    "ctranslate2",
    "pyannote",
    "speechbrain",
    "transformers",
    "tkinter",
    "matplotlib",
]

a = Analysis(  # noqa: F821
    [str(BACKEND / "packaging" / "entry.py")],
    pathex=[str(BACKEND / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    # UTF-8 mode внутри exe. Собранный интерпретатор НЕ читает PYTHONUTF8 из
    # окружения, а без него stdout в pipe идёт в кодировке системы (cp1251 /
    # cp1252), и первый же rich-print русского текста или «→» валит процесс
    # UnicodeEncodeError'ом.
    [("X utf8_mode=1", None, "OPTION")],
    exclude_binaries=True,
    name="twitch-cut-backend",
    debug=False,
    strip=False,
    upx=False,
    # Консольное приложение: Electron читает stdout (TWITCH_CUT_PORT=…) и
    # запускает процесс со скрытым окном (windowsHide).
    console=True,
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="twitch-cut-backend",
)
