#!/usr/bin/env bash
# Генерирует build/icon.icns из build/icon.ico.
# Запускать на macOS: bash desktop/scripts/make-icns.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD="$SCRIPT_DIR/../build"
ICONSET="$BUILD/icon.iconset"

mkdir -p "$ICONSET"

# Извлечь PNG из ico через sips (встроен в macOS).
sips -s format png "$BUILD/icon.ico" --out "$BUILD/icon_tmp.png" >/dev/null

# Нарезать нужные размеры для iconset.
for SIZE in 16 32 64 128 256 512; do
  sips -z $SIZE $SIZE "$BUILD/icon_tmp.png" --out "$ICONSET/icon_${SIZE}x${SIZE}.png" >/dev/null
  DOUBLE=$((SIZE * 2))
  sips -z $DOUBLE $DOUBLE "$BUILD/icon_tmp.png" --out "$ICONSET/icon_${SIZE}x${SIZE}@2x.png" >/dev/null
done

iconutil -c icns "$ICONSET" -o "$BUILD/icon.icns"

rm -rf "$ICONSET" "$BUILD/icon_tmp.png"
echo "Done: $BUILD/icon.icns"
