import { useCallback, useEffect, useState } from "react";
import {
  Cpu,
  FolderOpen,
  Info,
  Loader2,
  Mic,
  NotebookPen,
  RotateCcw,
  Save,
  SlidersHorizontal,
  Sparkles,
  Trash2,
} from "lucide-react";
import { Button } from "../components/ui/Button";
import { Card, CardDescription, CardTitle } from "../components/ui/Card";
import { Input, Label, Select } from "../components/ui/Input";
import { HelpTip } from "../components/ui/Tooltip";
import { toast } from "../components/ui/Toast";
import { getLogsDir, getSettings, putSettings, type Settings } from "../api";
import { AiProvidersCard } from "../components/settings/AiProvidersCard";

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
 * - Чувствительность к мату (default_sensitivity) — low | balanced | high; одна
 *   ручка вместо шести чисел мьюта, сами числа считает бэкенд.
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
  // Чувствительность детекции мата: low | balanced | high. Одна настройка на шесть
  // чисел бэкенда (отступы, склейка, поджим границ, порог спотинга) — сами числа
  // живут только в config.SENSITIVITY_LEVELS, сюда едет имя уровня.
  default_sensitivity: string;
  // WhisperX-only (используются, только если default_transcriber === "whisperx")
  default_model: string;
  default_language: string;
  default_batch_size: number;
  default_vad_method: string;
  default_vad_filter: boolean;
  // Переименовывать проект в настоящий тайтл стрима с Twitch при детекте оффтопа
  // (вместо имени папки vod_<id>). Тайтл тянется из get_vod_info по id.
  rename_project_to_title: boolean;
  // Журнал ручных правок: каждое сохранение из «Правки» дописывает, что человек
  // изменил относительно решения программы, в <decisions>.edits.json рядом с
  // проектом. Нужен для разбора «как должно быть vs как сделала программа»
  // (backend/src/twitch_cut/edit_journal.py, раздел в `twitch-cut audit`).
  edit_journal: boolean;
}

// Платформа хоста — от неё зависит, какой ускоритель имеет смысл: на
// Windows/Linux — GPU через DirectML/CUDA (cuda), на Mac — CPU.
const IS_MAC = window.twitchCut.platform === "darwin";

const DEFAULTS: KnownSettings = {
  default_transcriber: "gigaam",
  // На Mac — CPU (см. DEVICE_OPTIONS). На Windows/Linux — GPU: onnxruntime сам
  // уйдёт на CPU, если GPU нет.
  default_device: IS_MAC ? "cpu" : "cuda",
  default_compute_type: "float16",
  // balanced — единственная измеренная точка (см. config.SENSITIVITY_LEVELS):
  // выше неё F1 уже не растёт, ниже теряются находки.
  default_sensitivity: "balanced",
  default_model: "large-v3",
  default_language: "ru",
  default_batch_size: 16,
  default_vad_method: "pyannote",
  default_vad_filter: true,
  rename_project_to_title: true,
  // По умолчанию выключен: это диагностика, а не часть обычной работы.
  edit_journal: false,
};

// Списки опций — те же, что в NewJobScreen (держим синхронно вручную; вынесем
// в constants/, если появится третий потребитель).
const TRANSCRIBER_OPTIONS = [
  { value: "gigaam", label: "GigaAM v3 — лучший для русского" },
  { value: "whisperx", label: "WhisperX — универсальный, все языки" },
];
// Ускоритель зависит от платформы: на Windows/Linux — GPU через DirectML (любая
// видеокарта) или CUDA. На Mac выбирать нечего: CoreML с GigaAM не работает
// (см. onnx_models._providers_for), а CPU на Apple Silicon и так быстрый.
const DEVICE_OPTIONS = IS_MAC
  ? [{ value: "cpu", label: "cpu — на Apple Silicon быстро" }]
  : [
      { value: "cuda", label: "cuda — GPU (любая видеокарта через DirectML / NVIDIA CUDA)" },
      { value: "cpu", label: "cpu — работает везде, но медленнее" },
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
// Чувствительность: одна ручка «в какую сторону ошибаться». Подписи говорят про
// цену ошибки, а не про параметры — числа за ними стоят на бэке и замерены там же.
const SENSITIVITY_OPTIONS = [
  { value: "low", label: "Осторожно — не глушить лишнего" },
  { value: "balanced", label: "Обычно — рекомендуется" },
  { value: "high", label: "Строго — не пропустить мат" },
];

function pickKnown(raw: Settings): KnownSettings {
  const g = <K extends keyof KnownSettings>(k: K): KnownSettings[K] => {
    const v = raw[k];
    if (v == null) return DEFAULTS[k];
    // Runtime type-guard — на случай если файл правили руками.
    if (k === "default_vad_filter" || k === "rename_project_to_title" || k === "edit_journal") {
      return Boolean(v) as KnownSettings[K];
    }
    if (k === "default_batch_size") {
      const n = Number(v);
      return (Number.isFinite(n) ? n : DEFAULTS[k]) as KnownSettings[K];
    }
    // Уровень чувствительности — закрытый список. Мусор в файле (правили руками,
    // переехали уровни) отбрасываем здесь: иначе бэкенд отдал бы 422 на каждой
    // обработке, а причина была бы не видна.
    if (k === "default_sensitivity") {
      const s = String(v);
      const ok = SENSITIVITY_OPTIONS.some((o) => o.value === s);
      return (ok ? s : DEFAULTS[k]) as KnownSettings[K];
    }
    // Устройство — тоже закрытый список под платформу. У пользователей прошлых
    // версий на Mac сохранён 'mps', которого в списке больше нет (см.
    // DEVICE_OPTIONS) — показываем дефолт, а не пустой селект.
    if (k === "default_device") {
      const s = String(v);
      const ok = DEVICE_OPTIONS.some((o) => o.value === s);
      return (ok ? s : DEFAULTS[k]) as KnownSettings[K];
    }
    return String(v) as KnownSettings[K];
  };
  return {
    default_transcriber: g("default_transcriber"),
    default_device: g("default_device"),
    default_compute_type: g("default_compute_type"),
    default_sensitivity: g("default_sensitivity"),
    default_model: g("default_model"),
    default_language: g("default_language"),
    default_batch_size: g("default_batch_size"),
    default_vad_method: g("default_vad_method"),
    default_vad_filter: g("default_vad_filter"),
    rename_project_to_title: g("rename_project_to_title"),
    edit_journal: g("edit_journal"),
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
  const [section, setSection] = useState("recognition");
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

  const [wiping, setWiping] = useState<boolean>(false);

  const wipeData = async () => {
    // Нативный диалог подтверждения показывает main-процесс (dialog:wipe).
    // При успехе приложение перезапустится само — тост увидеть уже не успеем,
    // поэтому важна только ветка ошибки/отмены.
    setWiping(true);
    try {
      const res = await window.twitchCut.wipeData();
      if (res.cancelled) return;
      if (!res.ok) {
        toast.error("Не удалось удалить данные", {
          description: res.error ?? "Неизвестная ошибка",
        });
      }
    } catch (err) {
      toast.error("Не удалось удалить данные", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setWiping(false);
    }
  };

  // Версия: vite подставляет через define. См. vite.config.ts.
  const version = (import.meta as any).env?.VITE_APP_VERSION ?? "dev";

  const isWhisperx = known.default_transcriber === "whisperx";

  return (
    <div className="screen-page settings-page flex flex-col gap-5">
      <header className="flex items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-fg">Настройки</h1>
          <p className="mt-1 text-sm text-muted">
            Распознавание, подключённые модели и параметры приложения.
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

      <div className="settings-workspace">
        <nav className="settings-nav" aria-label="Разделы настроек">
          {[["recognition", "Распознавание"], ["providers", "Провайдеры ИИ"], ["projects", "Проекты и журнал"], ["application", "Приложение"]].map(([id, label]) => <button key={id} type="button" aria-current={section === id ? "page" : undefined} onClick={() => setSection(id)}>{label}</button>)}
        </nav>
        <div className="settings-content">
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

          <div hidden={section !== "recognition"} className="settings-group">
          {/* Движок распознавания — главный выбор проекта */}
          <Card variant="elevated">
            <div className="mb-4 flex items-start gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand/10 text-brand-from">
                <Mic className="h-4 w-4" />
              </div>
              <div>
                <CardTitle>Движок распознавания</CardTitle>
                <CardDescription>
                  <b>GigaAM v3</b> — для русской речи. <b>WhisperX</b> — для записей на разных языках.
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
                <p className="mt-1.5 text-xs text-subtle">
                  {known.default_transcriber === "gigaam"
                    ? "GigaAM — рекомендуется для русскоязычных стримов."
                    : "WhisperX — берите, если в записи не только русский."}
                </p>
              </div>
            </div>
          </Card>

          {/* Чувствительность — единственная ручка «в какую сторону ошибаться» */}
          <Card variant="elevated">
            <div className="mb-4 flex items-start gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand/10 text-brand-from">
                <SlidersHorizontal className="h-4 w-4" />
              </div>
              <div>
                <CardTitle>Чувствительность к мату</CardTitle>
                <CardDescription>
                  Определяет порог поиска и отступы заглушек. Более строгий режим может захватывать соседние слова.
                </CardDescription>
              </div>
            </div>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div className="md:col-span-2">
                <div className="mb-1.5 flex items-center gap-1.5">
                  <Label className="mb-0">Уровень</Label>
                  <HelpTip>
                    Замер на 15 минутах реального стрима (правки монтажёра как
                    истина): «Строго» глушит целиком 36 матов из 142 против 30 у
                    «Обычно», платя 3 лишними секундами мьюта на 15 минут.
                    «Осторожно» экономит 2 секунды, но теряет 3 попадания.
                  </HelpTip>
                </div>
                <Select
                  value={known.default_sensitivity}
                  onChange={(e) => set("default_sensitivity", e.target.value)}
                  options={SENSITIVITY_OPTIONS}
                />
                <p className="mt-1.5 text-xs text-subtle">
                  {known.default_sensitivity === "low"
                    ? "Мьюты уже — чистая речь почти не страдает, но хвост мата может остаться слышимым."
                    : known.default_sensitivity === "high"
                      ? "Мьюты шире и кандидатов на проверку больше — меньше шансов, что мат останется слышен."
                      : "Рекомендуется: измеренная точка равновесия, дальше точность не растёт."}
                </p>
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
                  {IS_MAC
                    ? "Для Apple Silicon используйте mps. CPU доступен на любом Mac."
                    : "Где считать: GPU (NVIDIA) или CPU. На машине без видеокарты выбирайте cpu — иначе обработка не запустится."}
                </CardDescription>
              </div>
            </div>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div>
                <div className="mb-1.5 flex items-center gap-1.5">
                  <Label className="mb-0">Устройство</Label>
                  <HelpTip>
                    {IS_MAC
                      ? "mps — GPU Apple (Metal), быстро на M1–M4. cpu — работает везде, но в разы медленнее."
                      : "cuda — GPU NVIDIA (нужны драйверы CUDA). cpu — работает везде, но в 10–30× медленнее."}
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
                  <Label className="mb-0">Точность вычислений</Label>
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

          {/* Провайдеры ИИ — ключи в keychain, конфиг в settings.json */}
          </div>
          <div hidden={section !== "providers"}><AiProvidersCard /></div>
          <div hidden={section !== "projects"} className="settings-group">

          {/* Именование проектов — переименование в настоящий тайтл с Twitch */}
          <Card variant="elevated">
            <div className="mb-4 flex items-start gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand/10 text-brand-from">
                <Sparkles className="h-4 w-4" />
              </div>
              <div>
                <CardTitle>Название проекта</CardTitle>
                <CardDescription>
                  При детекте оффтопа подтягивать настоящий тайтл стрима с Twitch
                  (по id из <code>vod_&lt;id&gt;</code>) и показывать его вместо
                  технического имени папки. Название хранится в проекте — файлы и
                  папки на диске не переименовываются.
                </CardDescription>
              </div>
            </div>
            <label className="flex items-center gap-2 text-sm text-muted">
              <input
                type="checkbox"
                checked={known.rename_project_to_title}
                onChange={(e) => set("rename_project_to_title", e.target.checked)}
                className="h-4 w-4 rounded border-white/10 bg-black/30 accent-brand-from"
              />
              Переименовывать проект в название стрима с Twitch (при детекте оффтопа)
            </label>
          </Card>

          {/* Журнал правок — диагностика качества детекции по ручным правкам */}
          <Card variant="elevated">
            <div className="mb-4 flex items-start gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand/10 text-brand-from">
                <NotebookPen className="h-4 w-4" />
              </div>
              <div>
                <CardTitle>Журнал правок</CardTitle>
                <CardDescription>
                  Записывать каждое сохранение из «Правки» в отдельный файл{" "}
                  <code>decisions.edits.json</code> рядом с проектом: что ты
                  добавил, снял или сдвинул относительно того, что решила
                  программа. Сам <code>decisions.json</code> не меняется.
                  <br />
                  Нужно, чтобы разобрать «как должно быть» против «как сделала
                  программа»: команда <code>twitch-cut audit</code> добавляет по
                  этому журналу раздел с диагнозом на каждую правку — где дыра в
                  словаре, где ASR услышал не то, где границы поехали.
                </CardDescription>
              </div>
            </div>
            <label className="flex items-center gap-2 text-sm text-muted">
              <input
                type="checkbox"
                checked={known.edit_journal}
                onChange={(e) => set("edit_journal", e.target.checked)}
                className="h-4 w-4 rounded border-white/10 bg-black/30 accent-brand-from"
              />
              Вести журнал ручных правок
            </label>
          </Card>

          </div>
          <div hidden={section !== "application"} className="settings-group">
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

          {/* Опасная зона — полное удаление скачанных данных (только macOS: на Windows есть деинсталлятор) */}
          {window.twitchCut.platform === "darwin" && <Card variant="surface" className="border-err/40 bg-err/5">
            <div className="mb-4 flex items-start gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-err/10 text-err">
                <Trash2 className="h-4 w-4" />
              </div>
              <div>
                <CardTitle>Полное удаление данных</CardTitle>
                <CardDescription>
                  Удалит скачанные Python, модели распознавания и виртуальное
                  окружение (~10 ГБ). Сама программа останется — её удаляют
                  отдельно через системный деинсталлятор. При следующем запуске
                  всё скачается заново.
                </CardDescription>
              </div>
            </div>
            <div className="flex items-center justify-between gap-4">
              <p className="text-xs text-muted">
                Пригодится перед удалением программы, чтобы не осталось «хвостов»
                на диске, или чтобы переустановить окружение с нуля.
              </p>
              <Button
                variant="destructive"
                size="md"
                onClick={wipeData}
                disabled={wiping}
                loading={wiping}
                className="shrink-0"
              >
                <Trash2 className="h-4 w-4" /> Удалить всё
              </Button>
            </div>
          </Card>}

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
          </div>
        </>
      )}
        </div>
      </div>
    </div>
  );
}
