import { useCallback, useEffect, useState } from "react";
import {
  Cpu,
  FolderOpen,
  Info,
  Loader2,
  RotateCcw,
  Save,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react";
import { Button } from "../components/ui/Button";
import { Card, CardDescription, CardTitle } from "../components/ui/Card";
import { Input, Label } from "../components/ui/Input";
import { toast } from "../components/ui/Toast";
import { getLogsDir, getSettings, putSettings, type Settings } from "../api";

/**
 * SettingsScreen — базовые настройки приложения.
 *
 * Что здесь:
 * - Defaults для формы NewJob (модель/язык/device/vad/batch_size/vad_filter).
 * - Кнопка «Открыть папку с конфигом» — там же лежат projects.json/settings.json.
 * - Версия из package.json (import.meta.env.VITE_APP_VERSION задаётся Vite'ом
 *   через define, см. vite.config.ts; fallback — читаем из runtime меты).
 *
 * Что НЕ здесь:
 * - Тема (пока только dark, toggle появится когда добавим light-палитру).
 * - Backend URL/порт (auto-detect через preload).
 * - Обновления (нет автообновлений).
 *
 * Хранение: raw dict в backend/settings.json — незнакомые поля сохраняются
 * round-trip. Форма показывает только известные ключи, unknown_keys видны
 * в expando "Дополнительно (сырой JSON)" — так фьючерные поля можно
 * подкрутить руками до релиза нового UI.
 */

// Известные ключи с их дефолтами. При сохранении отправляем весь объект
// (включая незнакомые поля из raw), а известным — присваиваем текущие
// значения полей формы.
interface KnownSettings {
  default_model: string;
  default_language: string;
  default_device: string;
  default_compute_type: string;
  default_batch_size: number;
  default_vad_method: string;
  default_vad_filter: boolean;
}

const DEFAULTS: KnownSettings = {
  default_model: "large-v3",
  default_language: "ru",
  default_device: "cuda",
  default_compute_type: "float16",
  default_batch_size: 16,
  default_vad_method: "pyannote",
  default_vad_filter: true,
};

function pickKnown(raw: Settings): KnownSettings {
  const g = <K extends keyof KnownSettings>(k: K): KnownSettings[K] => {
    const v = raw[k];
    if (v == null) return DEFAULTS[k];
    // Runtime type-guard для скалярных полей — на случай если файл правили
    // руками и написали "16" вместо 16 в batch_size.
    if (k === "default_vad_filter") return Boolean(v) as KnownSettings[K];
    if (k === "default_batch_size") {
      const n = Number(v);
      return (Number.isFinite(n) ? n : DEFAULTS[k]) as KnownSettings[K];
    }
    return String(v) as KnownSettings[K];
  };
  return {
    default_model: g("default_model"),
    default_language: g("default_language"),
    default_device: g("default_device"),
    default_compute_type: g("default_compute_type"),
    default_batch_size: g("default_batch_size"),
    default_vad_method: g("default_vad_method"),
    default_vad_filter: g("default_vad_filter"),
  };
}

function pickUnknown(raw: Settings): Settings {
  const known = new Set(Object.keys(DEFAULTS));
  const out: Settings = {};
  for (const [k, v] of Object.entries(raw)) {
    if (!known.has(k)) out[k] = v;
  }
  return out;
}

export function SettingsScreen() {
  const [loaded, setLoaded] = useState<boolean>(false);
  const [known, setKnown] = useState<KnownSettings>(DEFAULTS);
  const [unknown, setUnknown] = useState<Settings>({});
  const [dirty, setDirty] = useState<boolean>(false);
  const [saving, setSaving] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [logsPath, setLogsPath] = useState<string | null>(null);

  const set = useCallback(
    <K extends keyof KnownSettings>(k: K, v: KnownSettings[K]) => {
      setKnown((prev) => ({ ...prev, [k]: v }));
      setDirty(true);
    },
    []
  );

  const reload = useCallback(async () => {
    setError(null);
    try {
      const [raw, logs] = await Promise.all([getSettings(), getLogsDir()]);
      setKnown(pickKnown(raw));
      setUnknown(pickUnknown(raw));
      setLogsPath(logs.path);
      setDirty(false);
      setLoaded(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setLoaded(true);
    }
  }, []);

  useEffect(() => {
    reload();
  }, [reload]);

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      await putSettings({ ...unknown, ...known });
      setDirty(false);
      toast.success("Настройки сохранены");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      toast.error("Не удалось сохранить", { description: msg });
    } finally {
      setSaving(false);
    }
  };

  const resetDefaults = () => {
    setKnown(DEFAULTS);
    setDirty(true);
  };

  const openLogs = async () => {
    if (!logsPath) return;
    const ok = await window.twitchCut.openPath(logsPath);
    if (!ok) {
      toast.error("Не удалось открыть папку", { description: logsPath });
    }
  };

  // Версия: vite подставляет через define. См. vite.config.ts.
  const version =
    (import.meta as any).env?.VITE_APP_VERSION ?? "dev";

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-6 p-8">
      <header className="flex items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-fg">Настройки</h1>
          <p className="mt-1 text-sm text-muted">
            Дефолты для формы нового job'а и системные пути. Файл лежит рядом
            с реестром проектов.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            size="md"
            onClick={resetDefaults}
            disabled={!loaded || saving}
          >
            <RotateCcw className="h-4 w-4" /> Сбросить
          </Button>
          <Button onClick={save} disabled={!dirty || saving} loading={saving}>
            <Save className="h-4 w-4" /> Сохранить
          </Button>
        </div>
      </header>

      {!loaded ? (
        <div className="flex items-center gap-2 text-sm text-muted">
          <Loader2 className="h-4 w-4 animate-spin" /> Загрузка…
        </div>
      ) : (
        <>
          {error && (
            <Card variant="surface" className="border-err/40 bg-err/10 text-sm text-err">
              <div className="font-medium">Ошибка</div>
              <div className="mt-1 break-all font-mono-tabular text-[11px] text-err/90">
                {error}
              </div>
            </Card>
          )}

          {/* Транскрипция */}
          <Card variant="elevated">
            <div className="mb-4 flex items-start gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand/10 text-brand-from">
                <Sparkles className="h-4 w-4" />
              </div>
              <div>
                <CardTitle>Транскрипция (WhisperX)</CardTitle>
                <CardDescription>
                  Подставится в форму нового job'а. Можно переопределить на месте.
                </CardDescription>
              </div>
            </div>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div>
                <Label>Модель</Label>
                <Input
                  value={known.default_model}
                  onChange={(e) => set("default_model", e.target.value)}
                />
              </div>
              <div>
                <Label>Язык</Label>
                <Input
                  value={known.default_language}
                  onChange={(e) => set("default_language", e.target.value)}
                />
              </div>
            </div>
          </Card>

          {/* Устройство */}
          <Card variant="elevated">
            <div className="mb-4 flex items-start gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand/10 text-brand-from">
                <Cpu className="h-4 w-4" />
              </div>
              <div>
                <CardTitle>Вычисления</CardTitle>
                <CardDescription>Device, compute_type, batch_size.</CardDescription>
              </div>
            </div>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
              <div>
                <Label>Device</Label>
                <Input
                  value={known.default_device}
                  onChange={(e) => set("default_device", e.target.value)}
                  placeholder="cuda | cpu"
                />
              </div>
              <div>
                <Label>compute_type</Label>
                <Input
                  value={known.default_compute_type}
                  onChange={(e) => set("default_compute_type", e.target.value)}
                />
              </div>
              <div>
                <Label>batch_size</Label>
                <Input
                  value={String(known.default_batch_size)}
                  onChange={(e) => {
                    const n = Number(e.target.value);
                    set(
                      "default_batch_size",
                      Number.isFinite(n) && n > 0 ? n : DEFAULTS.default_batch_size
                    );
                  }}
                  inputMode="numeric"
                />
              </div>
            </div>
          </Card>

          {/* VAD */}
          <Card variant="elevated">
            <div className="mb-4 flex items-start gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand/10 text-brand-from">
                <SlidersHorizontal className="h-4 w-4" />
              </div>
              <div>
                <CardTitle>Voice Activity Detection</CardTitle>
                <CardDescription>
                  Метод и включён ли фильтр WhisperX по умолчанию.
                </CardDescription>
              </div>
            </div>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div>
                <Label>VAD method</Label>
                <Input
                  value={known.default_vad_method}
                  onChange={(e) => set("default_vad_method", e.target.value)}
                  placeholder="pyannote | silero"
                />
              </div>
              <label className="mt-6 flex items-center gap-2 text-sm text-muted">
                <input
                  type="checkbox"
                  checked={known.default_vad_filter}
                  onChange={(e) => set("default_vad_filter", e.target.checked)}
                  className="h-4 w-4 rounded border-white/10 bg-black/30 accent-brand-from"
                />
                Включить VAD-фильтр WhisperX
              </label>
            </div>
          </Card>

          {/* Информация / логи */}
          <Card variant="surface">
            <div className="mb-4 flex items-start gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-white/5 text-muted">
                <Info className="h-4 w-4" />
              </div>
              <div>
                <CardTitle>Приложение</CardTitle>
                <CardDescription>Пути и версия.</CardDescription>
              </div>
            </div>
            <dl className="grid grid-cols-[140px_1fr] items-center gap-x-4 gap-y-2 text-sm">
              <dt className="text-subtle">Версия</dt>
              <dd className="font-mono-tabular text-fg">{version}</dd>
              <dt className="text-subtle">Папка конфига</dt>
              <dd className="flex items-center gap-2">
                <code className="flex-1 truncate rounded bg-black/30 px-2 py-1 font-mono-tabular text-[11px] text-muted">
                  {logsPath ?? "…"}
                </code>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={openLogs}
                  disabled={!logsPath}
                >
                  <FolderOpen className="h-3.5 w-3.5" /> Открыть
                </Button>
              </dd>
            </dl>
          </Card>

          {/* Unknown keys — сырой JSON, чтобы не потерять фьючерные поля */}
          {Object.keys(unknown).length > 0 && (
            <Card variant="surface" padding="sm">
              <div className="mb-2 text-xs uppercase tracking-wider text-subtle">
                Дополнительные ключи (round-trip)
              </div>
              <pre className="overflow-x-auto rounded bg-black/40 p-3 font-mono-tabular text-[11px] text-muted">
                {JSON.stringify(unknown, null, 2)}
              </pre>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
