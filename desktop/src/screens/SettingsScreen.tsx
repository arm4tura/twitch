import { useCallback, useEffect, useState } from "react";
import {
  Cpu,
  FolderOpen,
  Info,
  Loader2,
  Mic,
  RotateCcw,
  Save,
  SlidersHorizontal,
} from "lucide-react";
import { Button } from "../components/ui/Button";
import { Card, CardDescription, CardTitle } from "../components/ui/Card";
import { Input, Label, Select } from "../components/ui/Input";
import { HelpTip } from "../components/ui/Tooltip";
import { toast } from "../components/ui/Toast";
import { getLogsDir, getSettings, putSettings, type Settings } from "../api";

/**
 * SettingsScreen — значения по умолчанию для новой обработки + системные пути.
 *
 * Проект по умолчанию работает на GigaAM v3 (русский, без CUDA-стека), поэтому
 * главная настройка здесь — ДВИЖОК распознавания. Параметры WhisperX (модель,
 * язык, VAD, batch) показываем только если выбран WhisperX — при GigaAM они не
 * применяются и не мозолят глаза.
 *
 * Что здесь:
 * - Движок распознавания (default_transcriber) — gigaam | whisperx.
 * - Вычисления: device + compute_type (единственное, что реально читает форма
 *   новой обработки на машине без GPU).
 * - Параметры WhisperX (только при whisperx): модель, язык, VAD, batch_size.
 * - Версия и папка конфига.
 *
 * Хранение: raw dict в backend/settings.json — незнакомые/скрытые поля
 * сохраняются round-trip (см. блок «Дополнительные ключи»). Убирая поле из
 * формы, мы НЕ удаляем его из файла.
 */

// Известные ключи с дефолтами. При сохранении шлём весь объект (unknown + known),
// известным присваиваем текущие значения формы.
interface KnownSettings {
  default_transcriber: string;
  default_device: string;
  default_compute_type: string;
  // WhisperX-only (используются, только если default_transcriber === "whisperx")
  default_model: string;
  default_language: string;
  default_batch_size: number;
  default_vad_method: string;
  default_vad_filter: boolean;
}

const DEFAULTS: KnownSettings = {
  default_transcriber: "gigaam",
  default_device: "cuda",
  default_compute_type: "float16",
  default_model: "large-v3",
  default_language: "ru",
  default_batch_size: 16,
  default_vad_method: "pyannote",
  default_vad_filter: true,
};

// Списки опций — те же, что в NewJobScreen (держим синхронно вручную; вынесем
// в constants/, если появится третий потребитель).
const TRANSCRIBER_OPTIONS = [
  { value: "gigaam", label: "GigaAM v3 — русский, по умолчанию" },
  { value: "whisperx", label: "WhisperX — мультиязычный, тонкая настройка" },
];
const DEVICE_OPTIONS = [
  { value: "cuda", label: "cuda — GPU (NVIDIA)" },
  { value: "cpu", label: "cpu — работает везде, но медленно" },
];
const COMPUTE_OPTIONS = [
  { value: "float16", label: "float16 — быстрее, нужно ≥ 8 GB VRAM" },
  { value: "int8", label: "int8 — экономно, годится для 4-6 GB VRAM" },
  { value: "float32", label: "float32 — максимум точности, медленно" },
];
const MODEL_OPTIONS = [
  { value: "tiny", label: "tiny — самая быстрая, низкое качество" },
  { value: "base", label: "base" },
  { value: "small", label: "small" },
  { value: "medium", label: "medium" },
  { value: "large-v2", label: "large-v2" },
  { value: "large-v3", label: "large-v3 — лучшее качество, медленно" },
];
const LANGUAGE_OPTIONS = [
  { value: "ru", label: "ru — русский" },
  { value: "en", label: "en — английский" },
  { value: "uk", label: "uk — украинский" },
  { value: "auto", label: "auto — автоопределение" },
];
const VAD_OPTIONS = [
  { value: "pyannote", label: "pyannote — точнее, но медленнее" },
  { value: "silero", label: "silero — быстрый VAD" },
];

function pickKnown(raw: Settings): KnownSettings {
  const g = <K extends keyof KnownSettings>(k: K): KnownSettings[K] => {
    const v = raw[k];
    if (v == null) return DEFAULTS[k];
    // Runtime type-guard — на случай если файл правили руками.
    if (k === "default_vad_filter") return Boolean(v) as KnownSettings[K];
    if (k === "default_batch_size") {
      const n = Number(v);
      return (Number.isFinite(n) ? n : DEFAULTS[k]) as KnownSettings[K];
    }
    return String(v) as KnownSettings[K];
  };
  return {
    default_transcriber: g("default_transcriber"),
    default_device: g("default_device"),
    default_compute_type: g("default_compute_type"),
    default_model: g("default_model"),
    default_language: g("default_language"),
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
  const version = (import.meta as any).env?.VITE_APP_VERSION ?? "dev";

  const isWhisperx = known.default_transcriber === "whisperx";

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-6 p-8">
      <header className="flex items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-fg">Настройки</h1>
          <p className="mt-1 text-sm text-muted">
            Значения по умолчанию для новой обработки и системные пути. Файл
            лежит рядом с реестром проектов.
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

          {/* Движок распознавания — главный выбор проекта */}
          <Card variant="elevated">
            <div className="mb-4 flex items-start gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand/10 text-brand-from">
                <Mic className="h-4 w-4" />
              </div>
              <div>
                <CardTitle>Движок распознавания</CardTitle>
                <CardDescription>
                  На чём распознавать речь по умолчанию. GigaAM v3 — русский, без
                  CUDA-стека; WhisperX — мультиязычный с тонкой настройкой.
                </CardDescription>
              </div>
            </div>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div className="md:col-span-2">
                <Label>Движок</Label>
                <Select
                  value={known.default_transcriber}
                  onChange={(e) => set("default_transcriber", e.target.value)}
                  options={TRANSCRIBER_OPTIONS}
                />
              </div>
            </div>
          </Card>

          {/* Вычисления — реально читается формой новой обработки */}
          <Card variant="elevated">
            <div className="mb-4 flex items-start gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand/10 text-brand-from">
                <Cpu className="h-4 w-4" />
              </div>
              <div>
                <CardTitle>Вычисления</CardTitle>
                <CardDescription>
                  Где считать: GPU (NVIDIA) или CPU. На машине без видеокарты
                  выбирайте cpu — иначе обработка не запустится.
                </CardDescription>
              </div>
            </div>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div>
                <div className="mb-1.5 flex items-center gap-1.5">
                  <Label className="mb-0">Device</Label>
                  <HelpTip>
                    cuda — GPU NVIDIA (нужны драйверы CUDA). cpu — работает
                    везде, но в 10–30× медленнее.
                  </HelpTip>
                </div>
                <Select
                  value={known.default_device}
                  onChange={(e) => set("default_device", e.target.value)}
                  options={DEVICE_OPTIONS}
                />
              </div>
              <div>
                <div className="mb-1.5 flex items-center gap-1.5">
                  <Label className="mb-0">compute_type</Label>
                  <HelpTip>
                    Точность вычислений. float16 — быстро, нужно 8+ GB VRAM.
                    int8 — влезает в 4–6 GB. float32 — эталон, для CPU или
                    большого GPU.
                  </HelpTip>
                </div>
                <Select
                  value={known.default_compute_type}
                  onChange={(e) => set("default_compute_type", e.target.value)}
                  options={COMPUTE_OPTIONS}
                />
              </div>
            </div>
          </Card>

          {/* Параметры WhisperX — только если выбран WhisperX */}
          {isWhisperx && (
            <Card variant="elevated">
              <div className="mb-4 flex items-start gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand/10 text-brand-from">
                  <SlidersHorizontal className="h-4 w-4" />
                </div>
                <div>
                  <CardTitle>Параметры WhisperX</CardTitle>
                  <CardDescription>
                    Подставятся в расширенном режиме новой обработки. Действуют
                    только для движка WhisperX.
                  </CardDescription>
                </div>
              </div>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <div>
                  <Label>Модель</Label>
                  <Select
                    value={known.default_model}
                    onChange={(e) => set("default_model", e.target.value)}
                    options={MODEL_OPTIONS}
                  />
                </div>
                <div>
                  <Label>Язык</Label>
                  <Select
                    value={known.default_language}
                    onChange={(e) => set("default_language", e.target.value)}
                    options={LANGUAGE_OPTIONS}
                  />
                </div>
                <div>
                  <Label>VAD method</Label>
                  <Select
                    value={known.default_vad_method}
                    onChange={(e) => set("default_vad_method", e.target.value)}
                    options={VAD_OPTIONS}
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
                <label className="flex items-center gap-2 text-sm text-muted md:col-span-2">
                  <input
                    type="checkbox"
                    checked={known.default_vad_filter}
                    onChange={(e) => set("default_vad_filter", e.target.checked)}
                    className="h-4 w-4 rounded border-white/10 bg-black/30 accent-brand-from"
                  />
                  Включить VAD-фильтр WhisperX по умолчанию
                </label>
              </div>
            </Card>
          )}

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

          {/* Unknown keys — сырой JSON, чтобы не потерять фьючерные/скрытые поля */}
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
