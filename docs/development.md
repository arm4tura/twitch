# Запуск из исходников и сборка

Раздел для разработчиков. Обычным пользователям это не нужно - им хватит [установщика](./getting-started.md).

## Что нужно

- Python 3.10-3.12 (сборка установщика - 3.12)
- Node.js 18+ и npm
- Видеокарта не обязательна: GigaAM идёт через onnxruntime (DirectML на Windows, CPU/CoreML на macOS)

## Backend (Python + распознавание)

Обычным пользователям ставить backend вручную не нужно - он собран
PyInstaller'ом и лежит внутри установщика, а при первом запуске приложение
докачивает только модели (`bootstrap.ts`). Ниже - установка CLI из репозитория
для разработки.

```bash
git clone https://github.com/arm4tura/twitch.git twitch && cd twitch/backend
python -m venv .venv
```

Активируйте venv и поставьте зависимости под свою платформу. torch в них нет:

```powershell
# Windows (onnxruntime-directml: GPU любой марки или CPU)
.\.venv\Scripts\Activate.ps1
pip install -r requirements-win.txt && pip install -e . --no-deps
```

```bash
# macOS / Linux
source .venv/bin/activate
pip install -r requirements-mac.txt && pip install -e . --no-deps
```

Затем скачайте модели и проверьте окружение:

```bash
twitch-cut prefetch   # ONNX-модели GigaAM + VAD, ~430 МБ
twitch-cut doctor     # проверка окружения
```

Необязательный движок WhisperX (torch + CUDA, несколько гигабайт):
`pip install -r requirements-whisperx-gpu.txt` (Windows + NVIDIA) или
`pip install -e ".[whisperx]"`, затем `twitch-cut prefetch --whisperx`.

### ONNX-модели

Распознавание и VAD работают на ONNX-моделях, которые получены из исходных
torch-чекпоинтов GigaAM v3 и pyannote segmentation-3.0. Пересобирать их нужно
только при смене модели:

```bash
pip install -e ".[export]"
pip install --no-deps git+https://github.com/salute-developers/GigaAM.git
python tools/export_onnx.py              # → ../models/onnx/*.onnx, печатает sha256
python tools/onnx_parity.py <аудио.wav>  # сверка с torch-путём: VAD, log-probs, текст
```

Новые файлы заливаются в релиз `models-v2` на GitHub, а их sha256 - в
`src/twitch_cut/onnx_models.py` (иначе `prefetch` их не примет).

## Desktop UI (Electron)

```bash
cd desktop
npm install
npm run dev          # Electron сам стартует twitch-cut serve из backend/.venv
```

## Сборка установщика

```bash
cd desktop
npm run dist            # соберёт backend (PyInstaller) и desktop/release/Twitch Cut Setup.exe
npm run dist:publish    # то же + заливка в GitHub Releases (нужен GH_TOKEN)
```

`predist` сам собирает бэкенд (`scripts/build-backend.js` → `backend/dist/<arch>/`)
Python'ом из `backend/.venv` - в нём должен стоять `pyinstaller`
(`pip install -e ".[build]"`). Кросс-сборки нет: установщик под архитектуру
собирается на машине этой архитектуры (в CI - отдельные раннеры для
Windows x64, macOS arm64 и macOS x64).

## Тесты

```bash
cd backend && python -m pytest
cd desktop && npm run typecheck
```

## Структура репозитория

```
twitch/
├─ backend/      # Python: CLI (twitch-cut) + сервер
│  ├─ src/twitch_cut/
│  ├─ tests/
│  └─ examples/  # тестовый транскрипт + smoke-скрипт
├─ desktop/      # Electron + React UI
│  ├─ electron/  # main / preload (Node-часть)
│  └─ src/       # интерфейс (экраны на React)
├─ models/       # кэш моделей распознавания
├─ work/         # прогоны: <basename>_<timestamp>/ с decisions.json, mutes.cs и кэшем
└─ docs/         # документация и скриншоты
```

### Данные: dev vs prod

`models/`, `work/` и реестр `projects.json` - это **данные**, а не код. Их
расположение задаёт `DATA_ROOT`:

- **В dev** (запуск из репо) `DATA_ROOT` = корень репозитория, поэтому `models/`
  и `work/` появляются прямо здесь. Оба каталога, а также `projects.json`,
  в `.gitignore` - в репозиторий не коммитятся.
- **В prod** (собранное приложение) сам код лежит в read-only каталоге, а данные
  Electron кладёт в writable-папку через `TWITCH_CUT_DATA_DIR` (userData или
  portable-папка рядом с exe). Тогда `models/`, `work/` и `projects.json` живут
  там, и portable-папка переносится целиком.

Отдельного `input/` для стрима нет: исходное видео выбирается файловым пикером
и может лежать где угодно. `output/` тоже нет - результаты (`decisions.json`,
`mutes.cs`) пишутся в папку прогона внутри `work/`.

## Стек

| Слой | Технологии |
|------|-----------|
| **Интерфейс** | Electron, React, Vite, TypeScript, TailwindCSS, Radix UI, wavesurfer.js |
| **Backend** | Python 3.10-3.12, FastAPI, uvicorn, Typer, ffmpeg |
| **Распознавание** | GigaAM v3 + pyannote VAD на onnxruntime (по умолчанию), WhisperX, faster-whisper, whisper.cpp |
| **Поиск мата** | pymorphy3, rapidfuzz, словарь banwords |
| **Экспорт** | скрипт `.cs` для Vegas Pro 21 |
