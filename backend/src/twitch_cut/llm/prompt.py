"""Промпт для NotebookLM и JSON Schema ожидаемого ответа.

NotebookLM не даёт structured output / tool use API, поэтому просим модель
вернуть **только** JSON в code-fence. Валидацию делает наша сторона в
importer.py — не полагаемся на «модель поняла».
"""

from __future__ import annotations

# Системная инструкция + пример. Не даём никаких примеров с несуществующими
# таймингами, чтобы не «якорить» модель на конкретные числа.
PROMPT_MARKDOWN = """\
# Задача

Ты — редактор нарезки реакции-стрима. В источниках лежит транскрипт стрима:
одна строка = одно слово с точным таймингом в формате `[HH:MM:SS.mmm] слово`.

Твоя задача — выбрать **{n_highlights}** самых ярких моментов длиной **30–90
секунд** каждый. «Яркими» считаются моменты, где происходит одно из:

- сильная эмоциональная реакция (испуг, шок, смех, восторг);
- меметичная реплика, которая хорошо звучит вне контекста;
- разбор/аналитика с чётким инсайтом (стример объясняет неочевидное);
- поворот сюжета в контенте, на который стример реагирует.

Не выбирай:

- пустые паузы / молчание;
- повторяющиеся звуки/бормотание;
- служебные фразы («так, погоди», «сейчас», «щас чекнем»);
- обсуждение донатов и технических моментов стрима.

# Формат ответа

Ответь **строго JSON-объектом** внутри одного code-fence с языком `json`.
Никакого текста до или после code-fence. Схема:

```json
{{
  "highlights": [
    {{
      "start_s": 123.456,
      "end_s": 178.900,
      "title": "короткое название (до 80 символов)",
      "reason": "почему этот момент яркий (2-4 предложения)",
      "score": 0.87,
      "quote": "короткая цитата из транскрипта для сверки"
    }}
  ]
}}
```

Правила:

- `start_s` и `end_s` — секунды из таймингов транскрипта, дробные, с
  точностью до миллисекунды. Округляй до реальных границ слов, а не до
  «красивых» чисел.
- `end_s - start_s` должно быть в диапазоне **30..90 секунд**.
- `score` — твоя уверенность в яркости от 0 до 1.
- `quote` — 5–15 слов подряд из выбранного фрагмента (для проверки).
- Верни ровно **{n_highlights}** highlights, отсортированные по `score`
  убыванию. Если ярких моментов меньше — верни столько, сколько нашёл, но
  не выдумывай.
- Все тайминги — в секундах от начала транскрипта (не HH:MM:SS).
"""


def build_prompt(n_highlights: int) -> str:
    if n_highlights < 1 or n_highlights > 50:
        raise ValueError("n_highlights must be in [1, 50]")
    return PROMPT_MARKDOWN.format(n_highlights=n_highlights)


# JSON Schema для документации / автогенерации UI-подсказки. НЕ используется
# для валидации: без внешнего jsonschema-пакета это лишний dep. Валидация —
# Pydantic-моделью Highlight в importer.py.
RESPONSE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["highlights"],
    "properties": {
        "highlights": {
            "type": "array",
            "minItems": 1,
            "maxItems": 50,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["start_s", "end_s", "title", "reason", "score"],
                "properties": {
                    "start_s": {"type": "number", "minimum": 0},
                    "end_s": {"type": "number", "exclusiveMinimum": 0},
                    "title": {"type": "string", "minLength": 1, "maxLength": 200},
                    "reason": {"type": "string", "minLength": 1, "maxLength": 2000},
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                    "quote": {"type": "string", "maxLength": 500},
                },
            },
        }
    },
}
