import { Check } from "lucide-react";
import { cn } from "../../lib/cn";

/**
 * Stepper — вертикальный индикатор стадий job'а.
 *
 * Design:
 * - Слева — колонка кружков-индикаторов, соединённых тонкой линией.
 * - Справа — заголовок стадии + короткое описание.
 * - Три состояния: done (заполненный brand + галочка), current (кольцо brand
 *   + пульсирующая точка), pending (пустой круг zinc).
 *
 * Стадии для процесса — жёсткий список из runner._run_process_sync:
 *   init → extract_audio → transcribe → detect_profanity → build_decisions → write_outputs
 * Смотрим stage-строку джобы и подсвечиваем «активную» + все предыдущие.
 */

export interface StepDef {
  key: string;
  title: string;
  description?: string;
  /** Список альтернативных backend-стадий, которые тоже маппятся сюда. */
  aliases?: string[];
}

export interface StepperProps {
  steps: StepDef[];
  /** Текущий backend-stage. Если не совпадает ни с чем — все считаются pending. */
  currentStage: string;
  /** Джоба завершена — все шаги отмечаются как done независимо от stage. */
  completed?: boolean;
  /** Джоба зафейлилась — currentStage помечается ошибкой, остальные — pending. */
  failed?: boolean;
  className?: string;
}

export function Stepper({
  steps,
  currentStage,
  completed,
  failed,
  className,
}: StepperProps) {
  const currentIdx = findStepIndex(steps, currentStage);

  return (
    <ol className={cn("relative flex flex-col gap-4", className)}>
      {steps.map((step, i) => {
        const state: "done" | "current" | "pending" | "failed" = completed
          ? "done"
          : failed && i === currentIdx
            ? "failed"
            : currentIdx === -1
              ? "pending"
              : i < currentIdx
                ? "done"
                : i === currentIdx
                  ? "current"
                  : "pending";
        const isLast = i === steps.length - 1;
        return (
          <li key={step.key} className="relative flex gap-3">
            {/* Соединительная линия */}
            {!isLast && (
              <span
                className={cn(
                  "absolute left-3 top-6 h-full w-px",
                  state === "done" ? "bg-brand-from/60" : "bg-white/8"
                )}
                aria-hidden
              />
            )}
            <StepDot state={state} />
            <div className="min-w-0 flex-1 pb-1">
              <div
                className={cn(
                  "text-sm font-medium leading-6",
                  state === "current" && "text-fg",
                  state === "done" && "text-muted",
                  state === "pending" && "text-subtle",
                  state === "failed" && "text-err"
                )}
              >
                {step.title}
              </div>
              {step.description && (
                <div className="text-xs text-subtle">{step.description}</div>
              )}
            </div>
          </li>
        );
      })}
    </ol>
  );
}

function StepDot({
  state,
}: {
  state: "done" | "current" | "pending" | "failed";
}) {
  return (
    <span
      className={cn(
        "relative z-10 mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full border transition-colors",
        state === "done" && "bg-brand border-brand-from text-white",
        state === "current" &&
          "border-brand-from bg-brand-from/10 shadow-[0_0_0_4px_rgb(139_92_246/0.12)]",
        state === "pending" && "border-white/15 bg-black/30",
        state === "failed" && "border-err bg-err/20 text-err"
      )}
    >
      {state === "done" && <Check className="h-3.5 w-3.5" />}
      {state === "current" && (
        <span className="h-2 w-2 rounded-full bg-brand-from animate-pulse-dot" />
      )}
      {state === "failed" && <span className="text-xs font-bold leading-none">!</span>}
    </span>
  );
}

function findStepIndex(steps: StepDef[], currentStage: string): number {
  if (!currentStage) return -1;
  const idx = steps.findIndex(
    (s) => s.key === currentStage || s.aliases?.includes(currentStage)
  );
  return idx;
}

/** Готовый список стадий для process job'а — под runner._run_process_sync. */
export const PROCESS_STEPS: StepDef[] = [
  {
    key: "init",
    title: "Подготовка",
    description: "Проверяем настройки и рабочую папку",
  },
  {
    key: "extract_audio",
    title: "Извлечение звука",
    description: "Достаём аудиодорожку из записи",
    aliases: ["mock_transcript"],
  },
  {
    key: "transcribe",
    title: "Распознавание речи",
    description: "Переводим речь в текст с таймингами",
  },
  {
    key: "detect_profanity",
    title: "Поиск мата",
    description: "Сверяем слова со словарём",
  },
  {
    key: "build_decisions",
    title: "Сборка правок",
    description: "Готовим список заглушек и вырезов",
  },
  {
    key: "write_outputs",
    title: "Сохранение",
    description: "Записываем результат и скрипт для Vegas",
  },
];

/** Стадии финального рендера — под runner._run_render_sync. */
export const RENDER_STEPS: StepDef[] = [
  {
    key: "init",
    title: "Подготовка",
    description: "Запуск рендера",
    aliases: ["prepare"],
  },
  {
    key: "render",
    title: "Монтаж видео",
    description: "Вырезаем сегменты и кодируем через ffmpeg",
  },
  {
    key: "done",
    title: "Готово",
    description: "Итоговый файл записан",
  },
];

/** Стадии скачивания VOD — под runner._run_download_vod_sync.
 * Backend эмитит prepare/info → download → verify → combine → done. `info` и
 * `prepare` (прогрев загрузчика) маппим в «Подготовку», чтобы не плодить шаги
 * ради пары секунд. */
export const DOWNLOAD_STEPS: StepDef[] = [
  {
    key: "prepare",
    title: "Подготовка",
    description: "Готовим загрузчик и получаем данные",
    aliases: ["init", "info"],
  },
  {
    key: "download",
    title: "Скачивание",
    description: "Качаем сегменты записи с Twitch",
  },
  {
    key: "verify",
    title: "Проверка",
    description: "Сверяем целостность сегментов",
  },
  {
    key: "combine",
    title: "Сборка",
    description: "Склеиваем сегменты в один файл",
  },
  {
    key: "done",
    title: "Готово",
    description: "Запись скачана",
  },
];

/** Стадии детекта тишины — под runner._run_detect_silence_sync. */
export const DETECT_SILENCE_STEPS: StepDef[] = [
  {
    key: "init",
    title: "Подготовка",
    description: "Запуск детекта",
    aliases: ["detect"],
  },
  {
    key: "postprocess",
    title: "Обработка пауз",
    description: "Склейка близких и добавление отступов",
  },
  {
    key: "done",
    title: "Готово",
    description: "Список пауз собран",
  },
];

/** Стадии AI-детекта оффтопа — под runner._run_detect_offtopic_sync.
 * Backend эмитит init/read → [read_source|transcribe_source] → classify →
 * merge → write. Транскрипция исходника (путь по источнику) — самая долгая
 * стадия (минуты ASR, 10–70%), поэтому у неё свой шаг; чтение готового
 * транскрипта (read_source) сворачиваем туда же. `read`/`init` и `merge` —
 * в «Подготовку»/«Запись», чтобы не плодить шаги ради секунд. */
export const DETECT_OFFTOPIC_STEPS: StepDef[] = [
  {
    key: "read",
    title: "Подготовка",
    description: "Читаем decisions и транскрипт",
    aliases: ["init"],
  },
  {
    key: "transcribe_source",
    title: "Транскрипция исходника",
    description: "Распознаём видео-первоисточник (ground truth)",
    aliases: ["read_source"],
  },
  {
    key: "classify",
    title: "Классификация",
    description: "LLM размечает оффтоп по репликам",
  },
  {
    key: "write",
    title: "Запись",
    description: "Мержим сегменты в decisions.json",
    aliases: ["merge"],
  },
];

/** Выбор набора стадий по типу job'а (JobState.kind). */
export function stepsForKind(kind: string | undefined): StepDef[] {
  switch (kind) {
    case "render":
      return RENDER_STEPS;
    case "detect_silence":
      return DETECT_SILENCE_STEPS;
    case "download_vod":
      return DOWNLOAD_STEPS;
    case "detect_offtopic":
      return DETECT_OFFTOPIC_STEPS;
    default:
      return PROCESS_STEPS;
  }
}

/** Человекочитаемый заголовок текущей стадии (по kind+stage, через aliases). */
export function stageLabel(
  kind: string | undefined,
  stage: string | undefined
): string {
  if (!stage) return "Обработка";
  const step = stepsForKind(kind).find(
    (s) => s.key === stage || s.aliases?.includes(stage)
  );
  return step?.title ?? "Обработка";
}
