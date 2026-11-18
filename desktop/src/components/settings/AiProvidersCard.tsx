import { useCallback, useEffect, useState } from "react";
import { Bot, CheckCircle2, Loader2, Plug, Plus, Trash2, XCircle } from "lucide-react";
import { Button } from "../ui/Button";
import { Card, CardDescription, CardTitle } from "../ui/Card";
import { Input, Label, Select } from "../ui/Input";
import { toast } from "../ui/Toast";
import {
  getAiProviders,
  getSettings,
  putSettings,
  testAiConnection,
  type AiProvider,
} from "../../api";

/**
 * AiProvidersCard — управление AI-провайдерами и их ключами (для оффтоп-детекции).
 *
 * Живёт отдельным компонентом со своим состоянием: в отличие от остального
 * SettingsScreen, ключи идут НЕ в settings.json, а в OS-keychain через IPC
 * (window.twitchCut.aiKey*), а конфиг кастомных провайдеров персистится
 * отдельным read-modify-write через getSettings/putSettings (ai_providers).
 *
 * БЕЗОПАСНОСТЬ (инвариант фазы): открытый ключ никогда не возвращается в
 * renderer. Мы вызываем aiKeySet(ref, key) чтобы СОХРАНИТЬ, aiKeyHas(ref) чтобы
 * узнать ЕСТЬ ЛИ ключ (без значения), и testAiConnection/{detect} — где main
 * сам достаёт ключ по ref и вставляет в loopback-запрос. Значение ключа после
 * сохранения в React-стейт не кладём (input очищаем).
 */

// Пустой драфт кастомного провайдера (openai-совместимый — самый частый кейс).
// models здесь только чтобы удовлетворить тип: при правке список моделей живёт
// в отдельном стейте modelsText (сырой текст), в массив его собирает saveDraft.
function emptyDraft(): AiProvider {
  return {
    id: "",
    label: "",
    format: "openai",
    base_url: "",
    models: [],
    api_key_ref: "",
    builtin: false,
  };
}

export function AiProvidersCard() {
  const [loading, setLoading] = useState(true);
  const [providers, setProviders] = useState<AiProvider[]>([]);
  const [selectedId, setSelectedId] = useState<string>("builtin:anthropic");
  const [keyInput, setKeyInput] = useState<string>("");
  const [savingKey, setSavingKey] = useState(false);
  const [testing, setTesting] = useState(false);
  // ref → есть ли сохранённый ключ (для бейджа «ключ сохранён»).
  const [keyPresence, setKeyPresence] = useState<Record<string, boolean>>({});

  const selected = providers.find((p) => p.id === selectedId) ?? null;
  // ref для ключа: у провайдера api_key_ref (у встроенных = их id).
  const keyRef = selected?.api_key_ref ?? null;
  const needsKey = keyRef !== null && keyRef !== undefined && keyRef !== "";
  const hasKey = keyRef ? Boolean(keyPresence[keyRef]) : false;

  const refreshPresence = useCallback(async (provs: AiProvider[]) => {
    const refs = Array.from(
      new Set(
        provs
          .map((p) => p.api_key_ref)
          .filter((r): r is string => typeof r === "string" && r.length > 0)
      )
    );
    const entries = await Promise.all(
      refs.map(async (ref) => {
        const res = await window.twitchCut.aiKeyHas(ref);
        return [ref, Boolean(res.ok && res.present)] as const;
      })
    );
    setKeyPresence(Object.fromEntries(entries));
  }, []);

  const reload = useCallback(async () => {
    setLoading(true);
    try {
      const { builtin, custom } = await getAiProviders();
      const all = [...builtin, ...custom];
      setProviders(all);
      setSelectedId((prev) => (all.some((p) => p.id === prev) ? prev : all[0]?.id ?? ""));
      await refreshPresence(all);
    } catch (err) {
      toast.error("Не удалось загрузить провайдеров ИИ", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setLoading(false);
    }
  }, [refreshPresence]);

  useEffect(() => {
    reload();
  }, [reload]);

  const saveKey = async () => {
    if (!keyRef || !keyInput.trim()) return;
    setSavingKey(true);
    try {
      const res = await window.twitchCut.aiKeySet(keyRef, keyInput.trim());
      if (!res.ok) {
        toast.error("Не удалось сохранить ключ", { description: res.error });
        return;
      }
      setKeyInput(""); // НИКОГДА не держим ключ в стейте после сохранения
      setKeyPresence((prev) => ({ ...prev, [keyRef]: true }));
      toast.success("Ключ сохранён в системном хранилище");
    } finally {
      setSavingKey(false);
    }
  };

  const deleteKey = async () => {
    if (!keyRef) return;
    const res = await window.twitchCut.aiKeyDelete(keyRef);
    if (!res.ok) {
      toast.error("Не удалось удалить ключ", { description: res.error });
      return;
    }
    setKeyPresence((prev) => ({ ...prev, [keyRef]: false }));
    toast.success("Ключ удалён");
  };

  const runTest = async () => {
    if (!selected) return;
    setTesting(true);
    try {
      const req = selected.builtin
        ? { provider_id: selected.id, api_key_ref: keyRef ?? undefined }
        : {
            provider_id: selected.id,
            provider: selected as unknown as Record<string, unknown>,
            api_key_ref: keyRef ?? undefined,
          };
      const res = await testAiConnection(req);
      if (res.ok) {
        toast.success("Соединение успешно", { description: res.detail });
      } else {
        toast.error("Провайдер недоступен", { description: res.detail });
      }
    } catch (err) {
      toast.error("Проверка не удалась", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setTesting(false);
    }
  };

  // --- кастомные провайдеры (persist в settings.json: ai_providers) ---------

  const [draft, setDraft] = useState<AiProvider | null>(null);
  // Список моделей редактируется как СЫРОЙ текст, а не как join(", ") массива:
  // иначе каждый рендер прогонял бы строку через split(",")→join(", ") и после
  // запятой на каждом нажатии клавиши накапливался бы лишний пробел
  // («a, b» → ["a"," b"] → «a,  b»). В массив строку превращает только
  // saveDraft — там же trim и выброс пустых. draft.models при правке не
  // используется (источник истины — этот стейт).
  const [modelsText, setModelsText] = useState("");

  const openDraft = () => {
    setDraft(emptyDraft());
    setModelsText("");
  };

  const closeDraft = () => {
    setDraft(null);
    setModelsText("");
  };

  const persistCustom = async (nextCustom: AiProvider[]) => {
    // Read-modify-write, чтобы не затереть остальные настройки.
    const raw = await getSettings();
    await putSettings({ ...raw, ai_providers: nextCustom });
  };

  const saveDraft = async () => {
    if (!draft) return;
    const id = draft.id.trim();
    const label = draft.label.trim();
    if (!id || !label) {
      toast.error("Заполните id и название провайдера");
      return;
    }
    if (id.startsWith("builtin:")) {
      toast.error("Префикс builtin: зарезервирован за встроенными");
      return;
    }
    if (draft.format === "openai" && !(draft.base_url ?? "").trim()) {
      toast.error("Для OpenAI-совместимого провайдера нужен base_url");
      return;
    }
    const custom = providers.filter((p) => !p.builtin);
    const cleaned: AiProvider = {
      id,
      label,
      format: draft.format,
      base_url: (draft.base_url ?? "").trim() || null,
      models: modelsText
        .split(",")
        .map((m) => m.trim())
        .filter((m) => m.length > 0),
      // api_key_ref = собственный id: одна запись ключа на провайдера.
      api_key_ref: id,
      builtin: false,
    };
    const next = [...custom.filter((p) => p.id !== id), cleaned];
    try {
      await persistCustom(next);
      toast.success("Провайдер сохранён");
      closeDraft();
      await reload();
      setSelectedId(id);
    } catch (err) {
      toast.error("Не удалось сохранить провайдера", {
        description: err instanceof Error ? err.message : String(err),
      });
    }
  };

  const deleteCustom = async (id: string) => {
    const custom = providers.filter((p) => !p.builtin && p.id !== id);
    try {
      await persistCustom(custom);
      // Ключ провайдера тоже подчищаем (ref == id для кастомных).
      await window.twitchCut.aiKeyDelete(id);
      toast.success("Провайдер удалён");
      await reload();
    } catch (err) {
      toast.error("Не удалось удалить провайдера", {
        description: err instanceof Error ? err.message : String(err),
      });
    }
  };

  const providerOptions = providers.map((p) => ({
    value: p.id,
    label: p.builtin ? p.label : `${p.label} (свой)`,
  }));

  return (
    <Card variant="elevated">
      <div className="mb-4 flex items-start gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand/10 text-brand-from">
          <Bot className="h-4 w-4" />
        </div>
        <div>
          <CardTitle>Провайдеры ИИ</CardTitle>
          <CardDescription>
            Для AI-детекции оффтопа на таймлайне. Ключ шифруется системным
            хранилищем (Keychain / DPAPI) и <b>никогда не пишется в настройки</b> —
            бэкенд получает его только на время запроса.
          </CardDescription>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center gap-2 text-sm text-muted">
          <Loader2 className="h-4 w-4 animate-spin" /> Загрузка…
        </div>
      ) : (
        <div className="flex flex-col gap-4">
          {/* Выбор активного провайдера */}
          <div>
            <Label>Провайдер</Label>
            <Select
              value={selectedId}
              onChange={(e) => setSelectedId(e.target.value)}
              options={providerOptions}
            />
            {selected && (
              <p className="mt-1.5 text-xs text-subtle">
                Формат: {selected.format}
                {selected.base_url ? ` · ${selected.base_url}` : ""}
                {selected.models.length > 0
                  ? ` · модели: ${selected.models.join(", ")}`
                  : ""}
              </p>
            )}
          </div>

          {/* Ключ провайдера */}
          {needsKey ? (
            <div className="rounded-lg border border-white/10 bg-black/20 p-3">
              <div className="mb-2 flex items-center justify-between gap-2">
                <Label className="mb-0">API-ключ</Label>
                {hasKey ? (
                  <span className="inline-flex items-center gap-1 text-xs text-ok">
                    <CheckCircle2 className="h-3.5 w-3.5" /> ключ сохранён
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 text-xs text-subtle">
                    <XCircle className="h-3.5 w-3.5" /> ключ не задан
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                <Input
                  type="password"
                  value={keyInput}
                  onChange={(e) => setKeyInput(e.target.value)}
                  placeholder={hasKey ? "•••••••• (введите, чтобы заменить)" : "Вставьте ключ"}
                  autoComplete="off"
                  className="flex-1"
                />
                <Button
                  variant="secondary"
                  size="md"
                  onClick={saveKey}
                  disabled={!keyInput.trim() || savingKey}
                  loading={savingKey}
                >
                  Сохранить
                </Button>
                {hasKey && (
                  <Button variant="ghost" size="icon" onClick={deleteKey} title="Удалить ключ">
                    <Trash2 className="h-4 w-4" />
                  </Button>
                )}
              </div>
              <div className="mt-3 flex items-center justify-between gap-2">
                <p className="text-xs text-subtle">
                  Ключ проверяется лёгким запросом — сам ключ не покидает
                  приложение (расшифровка в main-процессе).
                </p>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={runTest}
                  disabled={testing || (needsKey && !hasKey)}
                  loading={testing}
                  className="shrink-0"
                >
                  <Plug className="h-3.5 w-3.5" /> Проверить
                </Button>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-between gap-2 rounded-lg border border-white/10 bg-black/20 p-3">
              <p className="text-xs text-muted">
                Этот провайдер не требует ключа (например, локальная Ollama).
              </p>
              <Button
                variant="secondary"
                size="sm"
                onClick={runTest}
                disabled={testing}
                loading={testing}
              >
                <Plug className="h-3.5 w-3.5" /> Проверить
              </Button>
            </div>
          )}

          {/* Свои провайдеры: список + добавление */}
          <div className="border-t border-white/5 pt-3">
            <div className="mb-2 flex items-center justify-between">
              <div className="text-xs uppercase tracking-wider text-subtle">
                Свои провайдеры
              </div>
              {!draft && (
                <Button variant="ghost" size="sm" onClick={openDraft}>
                  <Plus className="h-3.5 w-3.5" /> Добавить
                </Button>
              )}
            </div>

            {providers.filter((p) => !p.builtin).length === 0 && !draft && (
              <p className="text-xs text-subtle">
                Пока нет своих провайдеров. Добавьте OpenAI-совместимый эндпоинт
                (OpenRouter, локальная модель и т.п.).
              </p>
            )}

            <ul className="flex flex-col gap-1">
              {providers
                .filter((p) => !p.builtin)
                .map((p) => (
                  <li
                    key={p.id}
                    className="flex items-center justify-between gap-2 rounded bg-black/20 px-2 py-1.5 text-sm"
                  >
                    <span className="truncate">
                      <span className="text-fg">{p.label}</span>{" "}
                      <span className="text-subtle">· {p.format}</span>
                    </span>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => deleteCustom(p.id)}
                      title="Удалить провайдера"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </li>
                ))}
            </ul>

            {draft && (
              <div className="mt-3 grid grid-cols-1 gap-3 rounded-lg border border-white/10 bg-black/20 p-3 md:grid-cols-2">
                <div>
                  <Label>id</Label>
                  <Input
                    value={draft.id}
                    onChange={(e) => setDraft({ ...draft, id: e.target.value })}
                    placeholder="my-openrouter"
                  />
                </div>
                <div>
                  <Label>Название</Label>
                  <Input
                    value={draft.label}
                    onChange={(e) => setDraft({ ...draft, label: e.target.value })}
                    placeholder="Мой OpenRouter"
                  />
                </div>
                <div>
                  <Label>Формат</Label>
                  <Select
                    value={draft.format}
                    onChange={(e) =>
                      setDraft({ ...draft, format: e.target.value as "anthropic" | "openai" })
                    }
                    options={[
                      { value: "openai", label: "openai — /v1/chat/completions" },
                      { value: "anthropic", label: "anthropic — Messages API" },
                    ]}
                  />
                </div>
                <div>
                  <Label>base_url</Label>
                  <Input
                    value={draft.base_url ?? ""}
                    onChange={(e) => setDraft({ ...draft, base_url: e.target.value })}
                    placeholder="https://openrouter.ai/api/v1"
                  />
                </div>
                <div className="md:col-span-2">
                  <Label>Модели (через запятую)</Label>
                  <Input
                    value={modelsText}
                    onChange={(e) => setModelsText(e.target.value)}
                    placeholder="anthropic/claude-opus-4.8, openai/gpt-4o-mini"
                  />
                </div>
                <div className="flex items-center gap-2 md:col-span-2">
                  <Button size="md" onClick={saveDraft}>
                    Сохранить провайдера
                  </Button>
                  <Button variant="ghost" size="md" onClick={closeDraft}>
                    Отмена
                  </Button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </Card>
  );
}
