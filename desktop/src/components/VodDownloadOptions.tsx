import { useMemo } from "react";
import { Scissors, Film, AlertCircle } from "lucide-react";
import { Card, CardTitle, CardDescription } from "./ui/Card";
import { Select, Label } from "./ui/Input";
import {
  TimecodeInput,
  timecodeToMs,
  msToTimecode,
} from "./ui/TimecodeInput";
import { fmtDurationMs } from "../lib/format";
import { cn } from "../lib/cn";

/**
 * VodDownloadOptions — карточка параметров скачивания VOD: выбор качества и
 * (опционально) обрезка по времени «с … по …».
 *
 * Обрезка ограничена длительностью VOD (`durationMs`):
 *   • «С» не может быть раньше 0 и позже «По»;
 *   • «По» не может быть позже конца записи;
 *   • оба поля — масковый TimecodeInput: нельзя ввести 3-значные минуты/секунды.
 *
 * Значения хранятся строками таймкода (как range_in/range_out в остальной форме),
 * чтобы бэкенд-контракт (HH:MM:SS) был единым.
 */

export interface VodQuality {
  /** Значение для CLI (напр. "1080p60", "source", "audio_only"). */
  value: string;
  /** Человекочитаемое (напр. "Source (1080p60)"). */
  label: string;
}

export interface VodDownloadOptionsProps {
  qualities: VodQuality[];
  quality: string;
  onQualityChange: (q: string) => void;

  /** Полная длительность VOD в мс (из info-режима CLI). null → неизвестна ещё. */
  durationMs: number | null;

  /** Включена ли обрезка. */
  trimEnabled: boolean;
  onTrimEnabledChange: (on: boolean) => void;

  /** Границы обрезки — строки таймкода. Пусто = с начала / до конца. */
  trimStart: string;
  trimEnd: string;
  onTrimStartChange: (v: string) => void;
  onTrimEndChange: (v: string) => void;
}

export function VodDownloadOptions({
  qualities,
  quality,
  onQualityChange,
  durationMs,
  trimEnabled,
  onTrimEnabledChange,
  trimStart,
  trimEnd,
  onTrimStartChange,
  onTrimEndChange,
}: VodDownloadOptionsProps) {
  const startMs = timecodeToMs(trimStart);
  const endMs = timecodeToMs(trimEnd);

  // Валидация диапазона для сообщения под полями. Клампинг делает сам
  // TimecodeInput (по maxMs/minMs), но перекрытие start<end проверяем тут:
  // maxMs у «С» = min(конец записи, введённое «По»); у «По» = конец записи.
  const rangeError = useMemo(() => {
    if (!trimEnabled) return null;
    if (durationMs == null) return null;
    if (startMs != null && startMs > durationMs)
      return "Начало позже конца записи.";
    if (endMs != null && endMs > durationMs)
      return `Конец не может быть позже ${msToTimecode(durationMs, false, 2)}.`;
    if (startMs != null && endMs != null && startMs >= endMs)
      return "«С» должно быть раньше «По».";
    return null;
  }, [trimEnabled, durationMs, startMs, endMs]);

  const selectedDurationMs = useMemo(() => {
    if (!trimEnabled) return durationMs;
    const from = startMs ?? 0;
    const to = endMs ?? durationMs ?? null;
    if (to == null) return null;
    return Math.max(0, to - from);
  }, [trimEnabled, startMs, endMs, durationMs]);

  return (
    <Card padding="sm" className="space-y-4">
      <div className="flex items-start gap-2">
        <Film className="mt-0.5 h-4 w-4 text-brand-from" />
        <div>
          <CardTitle className="text-sm">Параметры скачивания</CardTitle>
          <CardDescription className="text-xs">
            {durationMs != null
              ? `Длительность VOD — ${fmtDurationMs(durationMs)}`
              : "Определяю длительность…"}
          </CardDescription>
        </div>
      </div>

      {/* Качество */}
      <div>
        <Label>Качество</Label>
        <Select
          value={quality}
          onChange={(e) => onQualityChange(e.target.value)}
          options={
            qualities.length
              ? qualities.map((q) => ({ value: q.value, label: q.label }))
              : [{ value: "source", label: "Source (максимальное)" }]
          }
          disabled={!qualities.length}
        />
      </div>

      {/* Обрезка */}
      <div>
        <button
          type="button"
          onClick={() => onTrimEnabledChange(!trimEnabled)}
          className={cn(
            "flex items-center gap-1.5 text-sm transition-colors",
            trimEnabled ? "text-fg" : "text-muted hover:text-fg"
          )}
        >
          <Scissors className="h-3.5 w-3.5" />
          Скачать фрагмент
        </button>

        {trimEnabled && (
          <div className="mt-3 space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <Label className="text-xs">С</Label>
                <TimecodeInput
                  value={trimStart}
                  onChange={onTrimStartChange}
                  minMs={0}
                  // Начало не позже конца записи и не позже введённого «По».
                  maxMs={
                    endMs != null && durationMs != null
                      ? Math.min(endMs, durationMs)
                      : (endMs ?? durationMs ?? null)
                  }
                  placeholder="00:00:00"
                />
              </div>
              <div>
                <Label className="text-xs">По</Label>
                <TimecodeInput
                  value={trimEnd}
                  onChange={onTrimEndChange}
                  // Конец не раньше начала и не позже конца записи.
                  minMs={startMs ?? 0}
                  maxMs={durationMs}
                  placeholder={
                    durationMs != null
                      ? msToTimecode(durationMs, false, 2)
                      : "00:00:00"
                  }
                />
              </div>
            </div>

            {rangeError ? (
              <div className="flex items-center gap-1.5 text-xs text-err">
                <AlertCircle className="h-3.5 w-3.5" />
                {rangeError}
              </div>
            ) : selectedDurationMs != null ? (
              <div className="text-xs text-muted">
                К скачиванию — {fmtDurationMs(selectedDurationMs)}
              </div>
            ) : null}
          </div>
        )}
      </div>
    </Card>
  );
}
