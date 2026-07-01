<div align="center">

# 🎬 Twitch Reaction Cutter

**`v0.1.0`**

_Десктоп-приложение для полуавтоматической подготовки Twitch-реакций к монтажу в Vegas Pro 21._

`stream.mp4` → **WhisperX** → mute-маркеры → **Vegas `.cs` скрипт**

[Что это](#-что-это) · [Скриншот](#-как-выглядит) · [Установка](#-установка) · [Запуск](#-запуск) · [Как это работает](#-как-это-работает) · [Troubleshooting](#-troubleshooting) · [Лицензия](#-лицензия)

</div>

---

## ✨ Что это

**Twitch Reaction Cutter** — это десктопное приложение (Electron + React) поверх Python-бэкенда с WhisperX. Оно берёт длинный Twitch-VOD, находит мат по словарю и выдаёт **редактируемые mute-решения** + Vegas-скрипт, который расставит регионы и заглушит нужные куски.

Ничего не рендерит — только готовит монтажнику разметку.

Приложение состоит из двух частей, которые ставятся одним скриптом:

- **Desktop UI** (`desktop/`) — Electron + React + Tailwind. Импорт стрима, прогресс-бар этапов, таймлайн с waveform (wavesurfer.js), правка решений, экспорт.
- **Backend** (`backend/`) — Python CLI `twitch-cut` + FastAPI/WebSocket-сервер, который UI дёргает локально. Здесь живёт весь ASR/NLP.

```
┌───────────┐   ffmpeg    ┌───────────┐   WhisperX   ┌────────────┐   export   ┌────────────────┐
│ stream.mp4├────────────►│   WAV     ├─────────────►│ transcript ├───────────►│ decisions.json │
└───────────┘  16k mono   └───────────┘  RU align    └─────┬──────┘            │ vegas_build.cs │
                                                           │  banwords.txt     └────────────────┘
                                                           ▼
                                                      profanity map
```

## 🖼 Как выглядит

Desktop UI: дашборд с job'ами → New Job → прогресс этапов (extract / transcribe / detect / export) → Timeline с waveform и списком решений → Export → Vegas Pro 21.

## 🧱 Стек

| Слой | Технологии |
|------|-----------|
| **Desktop** | Electron 33 · React 18 · Vite 5 · TypeScript 5 · TailwindCSS · Radix UI · framer-motion · wavesurfer.js |
| **Backend** | Python 3.10–3.12 · Typer · FastAPI · uvicorn · WebSockets · ffmpeg |
| **ASR** | WhisperX `large-v3` · faster-whisper · CTranslate2 · pyannote.audio (+ whisper.cpp fallback) |
| **NLP** | pymorphy3 · rapidfuzz · RU banwords dictionary |
| **Export** | ScriptPortal.Vegas (`.cs` для Vegas Pro 21) |

## 📁 Структура

```
twitch/
├─ backend/          # Python: CLI (twitch-cut) + FastAPI-сервер
│  ├─ src/twitch_cut/
│  ├─ tests/
│  └─ examples/      # mock transcript + smoke script
├─ desktop/          # Electron + React UI
│  ├─ electron/      # main / preload (Node-часть)
│  └─ src/           # renderer (React screens)
├─ input/            # 📥 stream.mp4 · original_video.mp4
├─ models/           # 🧠 WhisperX / pyannote cache (~4 GB)
├─ work/             # 🛠  cache · checkpoints
├─ output/           # 📤 decisions.json · vegas_build.cs
├─ install.ps1       # Windows one-liner (backend)
├─ install.sh        # Linux one-liner (backend)
├─ docs/             # 📚 картинки и материалы для README
├─ LICENSE           # MIT
└─ README.md
```

## 🚀 Установка

> **Требования:** Windows 10/11 или Linux · Python 3.10–3.12 · NVIDIA GPU · CUDA ≥ 12.6 · Node.js ≥ 18 · npm · ~10 GB свободного места на диск

### 1. Backend (Python + WhisperX)

**Windows**
```powershell
git clone <this-repo> twitch && cd twitch
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

**Linux**
```bash
git clone <this-repo> twitch && cd twitch
bash ./install.sh
```

Скрипт проверит драйвер NVIDIA, создаст `backend/.venv`, поставит зависимости из `requirements-gpu.txt` (torch+cu126, whisperx, pyannote, faster-whisper и т.д.), запустит `doctor` и предложит скачать модели (~4 GB).

**Проверка окружения:**
```powershell
backend\.venv\Scripts\python -m twitch_cut.cli doctor
```

### 2. Desktop UI (Electron)

```bash
cd desktop
npm install
```

Всё — UI знает, где лежит `backend/.venv`, и сам поднимет FastAPI-сервер при старте.

## ▶️ Запуск

### Приложение (dev-режим)

```bash
cd desktop
npm run dev
```

Откроется окно Electron с UI. Electron сам стартует `twitch-cut serve` из `backend/.venv`, ловит порт и подключает renderer.

### Приложение (production build)

```bash
cd desktop
npm run build
npm start
```

### Внутри UI

1. **New Job** → выбери `input/stream.mp4` (опционально `original_video.mp4` для сравнения дорожек) и `banwords.txt`.
2. Дождись прогресса: `extract → transcribe → detect → export`. Каждый этап кэшируется в `work/<job>/cache/`.
3. **Timeline** — правишь mute-решения над waveform: reject/delete/disable, редактируешь границы.
4. **Export** — пересоберёт `output/vegas_build.cs` из текущего `decisions.json`.

### CLI (без UI)

Тот же бэкенд работает и как обычный CLI-инструмент:

```powershell
twitch-cut process `
  --stream    ..\input\stream.mp4 `
  --original  ..\input\original_video.mp4 `
  --banwords  .\banwords.txt `
  --workdir   ..\work\job_001 `
  --decisions ..\output\decisions.json `
  --vegas     ..\output\vegas_build.cs `
  --device cuda --compute-type float16
```

Без `--range-in / --range-out` обрабатывается всё видео. По умолчанию `--vad-method silero` — на Windows/CUDA это стабильнее pyannote.

### Smoke-тест (без GPU)

```powershell
python .\examples\smoke_mock.py
```

### Пересборка Vegas-скрипта после ручной правки `decisions.json`

```powershell
twitch-cut export --decisions ..\output\decisions.json --vegas ..\output\vegas_build.cs
```

## 🎞 Vegas Pro 21

1. Открой проект в Vegas Pro 21.
2. `Tools → Scripting → Run Script...` → выбери `output/vegas_build.cs`.
3. На таймлайне появятся регионы `MUTE profanity: ...`, аудио-события будут разрезаны и заглушены на границах слова.

Чтобы отклонить конкретный mute — поставь в `decisions.json` `"status": "rejected"` / `"deleted"` / `"disabled"` (или сделай это в UI) и пересобери скрипт.

## 🧠 Как это работает

1. **Extract** — ffmpeg вырезает WAV 16 kHz mono для нужного диапазона.
2. **Transcribe** — WhisperX (RU, `large-v3`) + forced alignment на пословный тайминг.
3. **Detect** — normalize (`ё→е`, lowercase) → pymorphy3 лемматизация → matching по `banwords.txt`.
4. **Extend** — mute тянется до конца Whisper-сегмента (естественной фразы) + padding, capped `--mute-max-seconds`.
5. **Export** — `decisions.json` (редактируемый) + `vegas_build.cs` для Vegas Pro 21.

Каждая стадия кэшируется в `work/<job>/cache/`. Пересчёт: `--force-extract` · `--force-transcribe` · `--force-detect`.

## 🔁 Alternative backend: whisper.cpp

Если PyTorch+CUDA на Windows не поднимается — используй whisper.cpp:

```powershell
twitch-cut process ... `
  --transcriber whispercpp `
  --whisper-cpp-bin   "C:\tools\whisper.cpp\whisper-cli.exe" `
  --whisper-cpp-model "C:\tools\whisper.cpp\models\ggml-large-v3.bin"
```

Свой CUDA-build, свой встроенный VAD, отдельный кэш (`transcript_whispercpp_*.json`).

## ✅ Тесты

```powershell
cd backend && python -m pytest
```

GPU не нужен, WhisperX не запускается.

```bash
cd desktop && npm run typecheck
```

## 📄 Лицензия

Проект распространяется под лицензией **MIT** — см. [`LICENSE`](./LICENSE).

Используемые библиотеки распространяются под совместимыми лицензиями:

| Библиотека | Лицензия |
|-----------|----------|
| WhisperX, faster-whisper | BSD-4-Clause / MIT |
| CTranslate2 | MIT |
| pyannote.audio | MIT |
| PyTorch, torchaudio, torchvision | BSD-3-Clause |
| transformers, huggingface_hub | Apache-2.0 |
| pymorphy3, pymorphy3-dicts-ru | MIT / LGPL |
| rapidfuzz | MIT |
| Typer, FastAPI, uvicorn, pydantic | MIT |
| Electron | MIT |
| React, React DOM | MIT |
| Vite, TailwindCSS, framer-motion | MIT |
| Radix UI, lucide-react, wavesurfer.js | MIT / BSD |

---

<div align="center">

## 💖 Поддержать разработку

Если проект сэкономил тебе часы монтажа — можно закинуть автору на кофе:

<a href="https://www.donationalerts.com/r/your_nick">
  <img src="./docs/assets/donate.png" alt="Donate on DonationAlerts" width="320" />
</a>

**[donationalerts.com/r/your_nick](https://www.donationalerts.com/r/your_nick)**

<sub>Made for editors who don't want to scrub 4 hours of stream by hand.</sub>

</div>
