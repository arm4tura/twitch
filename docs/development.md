# Запуск из исходников и сборка

Раздел для разработчиков. Обычным пользователям это не нужно - им хватит [установщика](./getting-started.md).

## Что нужно

- Python 3.10-3.12
- Node.js 18+ и npm
- Для GPU: NVIDIA + CUDA 12.6+

## Backend (Python + распознавание)

Обычным пользователям ставить backend вручную не нужно - desktop-приложение
поднимает окружение само (`bootstrap.ts`, авто-выбор CPU/GPU lock-файла). Ниже -
ручная установка CLI из репозитория для разработки.

```bash
git clone https://github.com/arm4tura/twitch.git twitch && cd twitch/backend
python -m venv .venv
```

Активируйте venv и поставьте зависимости под свою платформу:

```powershell
# Windows + NVIDIA GPU
.\.venv\Scripts\Activate.ps1
pip install -r requirements-gpu.txt && pip install -e . --no-deps
```

```powershell
# Windows без GPU (CPU)
.\.venv\Scripts\Activate.ps1
pip install -r requirements-cpu-win.txt && pip install -e . --no-deps
```

```bash
# macOS (CPU)
source .venv/bin/activate
pip install -r requirements-mac.txt && pip install -e . --no-deps
```

Затем скачайте модели и проверьте окружение:

```bash
twitch-cut prefetch   # ~4 GB моделей
twitch-cut doctor     # проверка окружения
```

## Desktop UI (Electron)

```bash
cd desktop
npm install
npm run dev          # Electron сам стартует twitch-cut serve из backend/.venv
```

## Сборка установщика

```bash
cd desktop
npm run dist            # соберёт desktop/release/Twitch Cut Setup.exe
npm run dist:publish    # то же + заливка в GitHub Releases (нужен GH_TOKEN)
```

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
| **Распознавание** | GigaAM v3 (по умолчанию), WhisperX, faster-whisper, whisper.cpp |
| **Поиск мата** | pymorphy3, rapidfuzz, словарь banwords |
| **Экспорт** | скрипт `.cs` для Vegas Pro 21 |
