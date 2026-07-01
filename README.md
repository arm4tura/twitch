<div align="center">

# 🎬 Twitch Reaction Cutter

**`v0.1.0`** — Phase 1 · CLI core

_Полуавтоматическая подготовка Twitch-реакций к монтажу в Vegas Pro 21._

`stream.mp4` → **WhisperX** → mute-маркеры → **Vegas `.cs` скрипт**

[Установка](#-установка) · [Запуск](#-запуск) · [Как это работает](#-как-это-работает) · [Troubleshooting](#-troubleshooting)

</div>

---

## ✨ Что делает

Берёт длинный Twitch-VOD, находит мат по словарю через WhisperX и выдаёт **редактируемые решения** + Vegas-скрипт, который расставит регионы и заглушит нужные куски. Ничего не рендерит, только готовит монтажнику разметку.

```
┌───────────┐   ffmpeg    ┌───────────┐   WhisperX   ┌────────────┐   export   ┌────────────────┐
│ stream.mp4├────────────►│   WAV     ├─────────────►│ transcript ├───────────►│ decisions.json │
└───────────┘  16k mono   └───────────┘  RU align    └─────┬──────┘            │ vegas_build.cs │
                                                           │  banwords.txt     └────────────────┘
                                                           ▼
                                                      profanity map
```

## 🧱 Стек

| Слой | Технологии |
|------|-----------|
| **Core** | Python 3.10–3.12 · Typer · ffmpeg |
| **ASR** | WhisperX `large-v3` (+ whisper.cpp fallback) |
| **NLP** | pymorphy3 · RU banwords dictionary |
| **Export** | ScriptPortal.Vegas (`.cs` для Vegas Pro 21) |
| **Desktop** _(WIP)_ | Electron · Vite · React · Tailwind |

## 📁 Структура

```
twitch/
├─ backend/          # Python CLI (twitch-cut)
│  ├─ src/twitch_cut/
│  ├─ tests/
│  └─ examples/      # mock transcript + smoke script
├─ desktop/          # Electron UI (placeholder)
├─ input/            # 📥 stream.mp4 · original_video.mp4
├─ models/           # 🧠 WhisperX / pyannote cache (~4 GB)
├─ work/             # 🛠  cache · checkpoints
├─ output/           # 📤 decisions.json · vegas_build.cs
├─ install.ps1       # Windows one-liner
└─ install.sh        # Linux one-liner
```

## 🚀 Установка

> **Требования:** Windows 10/11 или Linux · Python 3.10–3.12 · NVIDIA GPU · CUDA ≥ 12.6 · ~10 GB свободно

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

Скрипты проверят драйвер, создадут `backend/.venv`, поставят зависимости из `requirements-gpu.txt` (torch+cu126), запустят `doctor` и предложат скачать модели.

**Проверка окружения:**
```powershell
backend\.venv\Scripts\python -m twitch_cut.cli doctor
```

## ▶️ Запуск

### Реальный прогон

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

1. Откройте проект в Vegas Pro 21.
2. `Tools → Scripting → Run Script...` → выберите `output/vegas_build.cs`.
3. На таймлайне появятся регионы `MUTE profanity: ...`, аудио-события будут разрезаны и заглушены на границах слова.

Чтобы отклонить конкретный mute — поставьте в `decisions.json` `"status": "rejected"` / `"deleted"` / `"disabled"` и пересоберите скрипт.

## 🧠 Как это работает

1. **Extract** — ffmpeg вырезает WAV 16 kHz mono для нужного диапазона.
2. **Transcribe** — WhisperX (RU, `large-v3`) + forced alignment на пословный тайминг.
3. **Detect** — normalize (`ё→е`, lowercase) → pymorphy3 лемматизация → matching по `banwords.txt`.
4. **Extend** — mute тянется до конца Whisper-сегмента (естественной фразы) + padding, capped `--mute-max-seconds`.
5. **Export** — `decisions.json` (редактируемый) + `vegas_build.cs` для Vegas Pro 21.

Каждая стадия кэшируется в `work/<job>/cache/`. Пересчёт: `--force-extract` · `--force-transcribe` · `--force-detect`.

## 🔁 Alternative backend: whisper.cpp

Если PyTorch+CUDA на Windows не поднимается — используйте whisper.cpp:

```powershell
twitch-cut process ... `
  --transcriber whispercpp `
  --whisper-cpp-bin   "C:\tools\whisper.cpp\whisper-cli.exe" `
  --whisper-cpp-model "C:\tools\whisper.cpp\models\ggml-large-v3.bin"
```

Свой CUDA-build, свой встроенный VAD, отдельный кэш (`transcript_whispercpp_*.json`).

## 🩹 Troubleshooting

| Симптом | Что делать |
|--------|-----------|
| `Could not load symbol cudnnGetLibConfig` | `--vad-method silero --force-transcribe` |
| `FasterWhisperPipeline.transcribe() got unexpected 'vad_filter'` | обновите код, `--force-transcribe` |
| `torch` встал как CPU-версия | ставьте через `requirements-gpu.txt`, пакет — `pip install -e . --no-deps` |
| Всё падает — что вообще происходит | `twitch-cut doctor` |

## ✅ Тесты

```powershell
cd backend && python -m pytest
```

GPU не нужен, WhisperX не запускается.

## 🗺 Roadmap

- [x] **Phase 1** — CLI · WhisperX · mute-детекция · Vegas export
- [ ] **Phase 2** — sync с `original_video.mp4` · dead-air / AFK · vision
- [ ] **Phase 3** — NotebookLM для оффтопа · FastAPI/WebSocket
- [ ] **Phase 4** — Electron UI · `.exe` сборка · fuzzy matching

---

<div align="center">
<sub>Made for editors who don't want to scrub 4 hours of stream by hand.</sub>
</div>
