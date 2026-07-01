# =============================================================================
# twitch-reaction-cutter — Windows + NVIDIA GPU installer
#
# Использование (из корня репо):
#   powershell -ExecutionPolicy Bypass -File .\install.ps1
#
# Что делает:
#   1. Проверяет Python 3.10–3.12.
#   2. Проверяет nvidia-smi (нужна NVIDIA-карта с CUDA-драйвером).
#   3. Создаёт backend\.venv (если его ещё нет).
#   4. Ставит зависимости из backend\requirements-gpu.txt (torch+cu126).
#   5. Ставит сам пакет в editable-режиме.
#   6. Запускает `twitch-cut doctor` для проверки установки.
#   7. Опционально запускает `twitch-cut prefetch` (~4 GB моделей).
# =============================================================================

$ErrorActionPreference = "Stop"

# Выставляем UTF-8 для консоли, иначе кириллица в Write-Host ломается на
# cp866/cp1251 хостах (это чистая косметика — работе pip не мешает).
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
    chcp 65001 > $null
} catch {}

Write-Host ""
Write-Host "=== twitch-reaction-cutter installer (Windows + CUDA) ===" -ForegroundColor Cyan
Write-Host ""

# --- 1. Python -----------------------------------------------------------------
# Ищем интерпретатор 3.10-3.12. Пробуем `python`, потом `py -3.12` и т.д.
# Не используем `2>&1` — на Windows PowerShell 5.1 stderr-объекты типа
# ErrorRecord ломают string-parsing через $matches.
function Get-PythonVersion($cmd, $args) {
    try {
        $out = & $cmd $args --version 2>$null
        if ($LASTEXITCODE -eq 0 -and $out -match "Python (\d+)\.(\d+)\.(\d+)") {
            return @{ Major = [int]$matches[1]; Minor = [int]$matches[2]; Version = $out }
        }
    } catch {}
    return $null
}

$pythonCmd = $null
$pythonArgs = @()

$candidates = @(
    @{Cmd="python"; Args=@()},
    @{Cmd="py";     Args=@("-3.12")},
    @{Cmd="py";     Args=@("-3.11")},
    @{Cmd="py";     Args=@("-3.10")}
)
foreach ($c in $candidates) {
    $info = Get-PythonVersion $c.Cmd $c.Args
    if ($info -and $info.Major -eq 3 -and $info.Minor -ge 10 -and $info.Minor -le 12) {
        $pythonCmd = $c.Cmd
        $pythonArgs = $c.Args
        Write-Host "[OK]   $($info.Version) ($($c.Cmd) $($c.Args -join ' '))" -ForegroundColor Green
        break
    }
}
if (-not $pythonCmd) {
    Write-Host "[FAIL] Не найден Python 3.10, 3.11 или 3.12." -ForegroundColor Red
    Write-Host "       Скачайте с https://www.python.org/downloads/ и повторите." -ForegroundColor Red
    exit 1
}

# --- 2. NVIDIA / CUDA driver ---------------------------------------------------
# Проверяем наличие nvidia-smi.exe в PATH и его exit code — без `2>&1`,
# который на 5.1 превращает stderr в ErrorRecord и триггерит false-positive
# в try/catch (именно так упало у пользователя: nvidia-smi работал, а мы
# ловили исключение).
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $nvidiaSmi) {
    Write-Host "[FAIL] nvidia-smi не найден в PATH — нет NVIDIA-драйвера." -ForegroundColor Red
    Write-Host "       Установите драйвер с https://www.nvidia.com/drivers" -ForegroundColor Red
    exit 1
}

$nvsmiOutput = & nvidia-smi
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] nvidia-smi вернул код $LASTEXITCODE." -ForegroundColor Red
    exit 1
}
# Ищем "CUDA Version: 12.x" или "CUDA UMD Version: 13.x" в шапке.
$cudaLine = ($nvsmiOutput | Select-String "CUDA (UMD )?Version" | Select-Object -First 1)
if ($cudaLine) {
    Write-Host "[OK]   NVIDIA-драйвер: $($cudaLine.ToString().Trim())" -ForegroundColor Green
} else {
    Write-Host "[OK]   NVIDIA-драйвер найден (CUDA-строку в шапке не распознали, но nvidia-smi работает)." -ForegroundColor Green
}

# --- 3. .venv ------------------------------------------------------------------
$repoRoot  = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend   = Join-Path $repoRoot "backend"
$venvDir   = Join-Path $backend ".venv"
$venvPy    = Join-Path $venvDir "Scripts\python.exe"

if (Test-Path $venvPy) {
    Write-Host "[OK]   .venv уже существует: $venvDir" -ForegroundColor Green
} else {
    Write-Host "[..]   Создаю .venv в $venvDir ..." -ForegroundColor Yellow
    & $pythonCmd @pythonArgs -m venv $venvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] Не удалось создать .venv." -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK]   .venv создан." -ForegroundColor Green
}

# --- 4. pip upgrade + requirements-gpu.txt -------------------------------------
Write-Host "[..]   Обновляю pip ..." -ForegroundColor Yellow
& $venvPy -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { Write-Host "[FAIL] pip upgrade" -ForegroundColor Red; exit 1 }

$reqFile = Join-Path $backend "requirements-gpu.txt"
Write-Host "[..]   Ставлю зависимости из $reqFile (~5-10 минут, качается torch+cu126)..." -ForegroundColor Yellow
& $venvPy -m pip install -r $reqFile
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Установка requirements-gpu.txt провалилась." -ForegroundColor Red
    exit 1
}
Write-Host "[OK]   Зависимости установлены." -ForegroundColor Green

# --- 5. editable install пакета ------------------------------------------------
Write-Host "[..]   Ставлю twitch-reaction-cutter в editable-режиме ..." -ForegroundColor Yellow
Push-Location $backend
try {
    & $venvPy -m pip install -e . --no-deps
} finally {
    Pop-Location
}
if ($LASTEXITCODE -ne 0) { Write-Host "[FAIL] pip install -e ." -ForegroundColor Red; exit 1 }
Write-Host "[OK]   Пакет установлен." -ForegroundColor Green

# --- 6.5. desktop UI (Electron + React) — опционально ------------------------
# Если Node.js есть, собираем фронт и Electron main. Без Node.js — просто
# пропускаем; CLI (`twitch-cut process/export/…`) полностью работоспособен.
$desktopDir = Join-Path $PSScriptRoot "desktop"
$nodeCmd = Get-Command node -ErrorAction SilentlyContinue
if ((Test-Path $desktopDir) -and $nodeCmd) {
    Write-Host ""
    Write-Host "=== 6.5. Сборка desktop UI ===" -ForegroundColor Cyan
    Write-Host "[..]   node $(& node --version) — ставлю npm-зависимости ..." -ForegroundColor Yellow
    Push-Location $desktopDir
    try {
        # `npm ci` требует package-lock.json — в первой установке его может не
        # быть, fallback на `npm install`.
        if (Test-Path (Join-Path $desktopDir "package-lock.json")) {
            & npm ci
        } else {
            & npm install
        }
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[WARN] npm install упал — desktop UI не собран. CLI работать будет." -ForegroundColor Yellow
        } else {
            Write-Host "[..]   npm run build ..." -ForegroundColor Yellow
            & npm run build
            if ($LASTEXITCODE -ne 0) {
                Write-Host "[WARN] npm run build упал — desktop UI не собран." -ForegroundColor Yellow
            } else {
                Write-Host "[OK]   desktop UI собран → desktop\dist\" -ForegroundColor Green
            }
        }
    } finally {
        Pop-Location
    }
} elseif (Test-Path $desktopDir) {
    Write-Host ""
    Write-Host "[WARN] Node.js не найден — desktop UI не собран." -ForegroundColor Yellow
    Write-Host "       Установи Node.js LTS с https://nodejs.org и перезапусти install.ps1" -ForegroundColor Yellow
    Write-Host "       (CLI работает без Node.js — UI это опциональный слой поверх backend)" -ForegroundColor Yellow
}

# --- 7. doctor -----------------------------------------------------------------
Write-Host ""
Write-Host "=== doctor: проверка окружения ===" -ForegroundColor Cyan
& $venvPy -m twitch_cut.cli doctor
$doctorExit = $LASTEXITCODE

# --- 7. prefetch (опционально) -------------------------------------------------
if ($doctorExit -eq 0) {
    Write-Host ""
    $answer = Read-Host "Скачать модели сейчас? Это ~4 GB и займёт время. [y/N]"
    if ($answer -match "^[Yy]") {
        & $venvPy -m twitch_cut.cli prefetch
    } else {
        Write-Host "Пропущено. Запусти позже:  .venv\Scripts\python -m twitch_cut.cli prefetch" -ForegroundColor Yellow
    }
}

# --- Готово --------------------------------------------------------------------
Write-Host ""
Write-Host "=== Установка завершена ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Активируй venv:      backend\.venv\Scripts\Activate.ps1"
Write-Host "Или запусти напрямую: backend\.venv\Scripts\python -m twitch_cut.cli process --help"
Write-Host ""

if ($doctorExit -ne 0) {
    Write-Host "ВНИМАНИЕ: doctor нашёл проблемы (см. выше)." -ForegroundColor Yellow
    exit $doctorExit
}
