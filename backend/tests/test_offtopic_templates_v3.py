"""Фаза 3: новые виды шаблонов не ломают старые и валидируются как надо.

Ключевое требование плана: старые встроенные шаблоны read-only, на них могут
ссылаться уже сделанные пользователем клоны, поэтому их id/вид/плейсхолдеры не
должны меняться при добавлении пути по исходнику.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from twitch_cut.llm_offtopic.templates import (
    BUILTIN_TEMPLATES,
    PromptTemplate,
    TemplateKind,
    builtin_template,
    builtin_template_for_kind,
)


def test_legacy_builtins_are_unchanged():
    # Точные id и виды — контракт для клонов. Меняем список только ДОБАВЛЕНИЕМ.
    topic = builtin_template("builtin:topic-ru")
    classification = builtin_template("builtin:classification-ru")
    assert topic is not None and topic.kind is TemplateKind.TOPIC
    assert topic.placeholders() == {"transcript"}
    assert classification is not None and classification.kind is TemplateKind.CLASSIFICATION
    assert classification.placeholders() == {"topic", "segment"}


@pytest.mark.parametrize(
    "kind, placeholders",
    [
        (TemplateKind.SOURCE_OUTLINE, {"transcript"}),
        (TemplateKind.SOURCE_CLASSIFICATION, {"outline", "turns"}),
        (TemplateKind.SINGLE_PASS, {"transcript"}),
    ],
)
def test_new_builtins_exist_with_expected_placeholders(kind, placeholders):
    tpl = builtin_template_for_kind(kind)
    assert tpl.kind is kind
    assert tpl.placeholders() == placeholders
    assert tpl.builtin is True


def test_source_classification_keeps_two_mandatory_rules():
    # Без этих двух правил путь по исходнику ломается известным образом:
    # совпадение по ключевому слову засчитывалось как «по теме», а любая пауза —
    # как оффтоп. Правила зашиты в built-in — фиксируем их присутствие.
    text = builtin_template_for_kind(TemplateKind.SOURCE_CLASSIFICATION).text
    assert "по ключевому слову" in text  # правило 1: keyword-overlap ≠ по теме
    assert "комментирует" in text and "КАДР" in text  # критерий «кадр или своё»
    assert "короткая пауза" in text.lower() or "короткая пауза сама" in text  # правило 2


@pytest.mark.parametrize("kind", list(TemplateKind))
def test_missing_required_placeholder_is_rejected(kind):
    with pytest.raises(ValidationError):
        PromptTemplate(id="x", label="x", kind=kind, text="совсем без плейсхолдеров")


def test_extra_placeholder_is_rejected():
    # Лишний {name} = KeyError при .format в рантайме — ловим на сохранении.
    with pytest.raises(ValidationError):
        PromptTemplate(
            id="x",
            label="x",
            kind=TemplateKind.SOURCE_CLASSIFICATION,
            text="{outline} {turns} {lishnee}",
        )
