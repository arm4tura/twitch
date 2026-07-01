#!/usr/bin/env bash
# =============================================================================
# twitch-reaction-cutter — Linux + NVIDIA GPU installer
# (Windows: используйте install.ps1)
#
# Использование (из корня репо):
#   bash ./install.sh
# =============================================================================

set -euo pipefail

echo ""
echo "=== twitch-reaction-cutter installer (Linux + CUDA) ==="
echo ""

# --- 1. Python -----------------------------------------------------------------
PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        ver=$("$candidate" --version 2>&1)
        if echo "$ver" | grep -qE "Python 3\.(1[0-2])\.[0-9]+"; then
            PYTHON="$candidate"
            echo "[OK]   $ver ($candidate)"
            break
        fi
    fi
done
if [ -z "$PYTHON" ]; then
    echo "[FAIL] Нужен Python 3.10, 3.11 или 3.12."
    exit 1
fi

# --- 2. NVIDIA / CUDA driver ---------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[FAIL] nvidia-smi не найден — нет NVIDIA-драйвера."
    echo "       Проект требует NVIDIA GPU + CUDA >= 12.6."
    exit 1
fi
cuda_line=$(nvidia-smi | grep -m1 "CUDA Version" || true)
echo "[OK]   $cuda_line"

# --- 3. .venv ------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$REPO_ROOT/backend"
VENV_DIR="$BACKEND/.venv"
VENV_PY="$VENV_DIR/bin/python"

if [ -x "$VENV_PY" ]; then
    echo "[OK]   .venv уже существует: $VENV_DIR"
else
    echo "[..]   Создаю .venv в $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
    echo "[OK]   .venv создан."
fi

# --- 4. pip + requirements-gpu.txt ---------------------------------------------
echo "[..]   Обновляю pip ..."
"$VENV_PY" -m pip install --upgrade pip

REQ_FILE="$BACKEND/requirements-gpu.txt"
echo "[..]   Ставлю зависимости из $REQ_FILE (~5-10 минут, качается torch+cu126)..."
"$VENV_PY" -m pip install -r "$REQ_FILE"
echo "[OK]   Зависимости установлены."

# --- 5. editable install -------------------------------------------------------
echo "[..]   Ставлю twitch-reaction-cutter в editable-режиме ..."
(cd "$BACKEND" && "$VENV_PY" -m pip install -e . --no-deps)
echo "[OK]   Пакет установлен."

# --- 6. doctor -----------------------------------------------------------------
echo ""
echo "=== doctor: проверка окружения ==="
if "$VENV_PY" -m twitch_cut.cli doctor; then
    DOCTOR_OK=1
else
    DOCTOR_OK=0
fi

# --- 7. prefetch ---------------------------------------------------------------
if [ "$DOCTOR_OK" -eq 1 ]; then
    echo ""
    read -rp "Скачать модели сейчас? ~4 GB. [y/N] " answer
    if [[ "$answer" =~ ^[Yy] ]]; then
        "$VENV_PY" -m twitch_cut.cli prefetch
    else
        echo "Пропущено. Запусти позже:  $VENV_PY -m twitch_cut.cli prefetch"
    fi
fi

echo ""
echo "=== Установка завершена ==="
echo "Запуск:  source $VENV_DIR/bin/activate  && twitch-cut process --help"
echo ""

if [ "$DOCTOR_OK" -eq 0 ]; then
    echo "ВНИМАНИЕ: doctor нашёл проблемы (см. выше)."
    exit 1
fi
