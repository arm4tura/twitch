"""Тесты monkey-patch'а speechbrain LazyModule из transcription.py.

РЕГРЕССИЯ, которую они ловят: обёртка `_safe_getattr` обязана звать
`self.ensure_module(1)` НАПРЯМУЮ. Внутри `ensure_module` speechbrain смотрит
`sys._getframe(stacklevel + 1)` и, если вызывающий фрейм — stdlib `inspect.py`,
поднимает AttributeError, чтобы НЕ делать ленивый импорт (защита от
`inspect.getmodule()`, который перебирает `sys.modules` и дёргает `__file__`
у каждого модуля).

Если обёртка вызывает оригинальный `__getattr__`, добавляется лишний
Python-фрейм, счёт `_getframe` съезжает, guard не срабатывает — и speechbrain
реально импортирует модуль во время обхода стека. Симптомы:
  - шумный `UserWarning: Module 'speechbrain.pretrained' was deprecated`,
    атрибутированный на transcription.py вместо кода пользователя;
  - в глубоком стеке `getframeinfo` → `getmodule` → снова `__getattr__`
    по всем LazyModule → RecursionError.
"""
from __future__ import annotations

import inspect
import sys
import warnings

import pytest

from twitch_cut.transcription import _stub_missing_speechbrain_optionals

pytest.importorskip("speechbrain", reason="speechbrain не установлен в этом окружении")


@pytest.fixture(scope="module", autouse=True)
def _patched() -> None:
    """Патч идемпотентен и глобален, ставим один раз на модуль."""
    import speechbrain  # noqa: F401  — регистрирует deprecated-redirect'ы

    _stub_missing_speechbrain_optionals()


def _lazy_deprecated_module():
    mod = sys.modules.get("speechbrain.pretrained")
    if mod is None or getattr(mod, "lazy_module", "missing") == "missing":
        pytest.skip("speechbrain не регистрирует LazyModule для speechbrain.pretrained")
    return mod


def test_inspect_guard_survives_patch() -> None:
    """getattr из фрейма inspect.py НЕ должен триггерить ленивый импорт."""
    mod = _lazy_deprecated_module()
    if mod.lazy_module is not None:
        pytest.skip("speechbrain.inference уже импортирован другим тестом")

    # Компилируем с именем файла stdlib inspect.py — ровно так guard и решает,
    # что обращение идёт из служебного обхода sys.modules, а не из кода юзера.
    code = compile(
        "result = getattr(lazy_mod, '__file__', SENTINEL)",
        "/usr/lib/python3.10/inspect.py",
        "exec",
    )
    namespace: dict = {"lazy_mod": mod, "SENTINEL": "<<default>>"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exec(code, namespace)  # noqa: S102 — контролируемый литерал, нужен fake-filename

    assert namespace["result"] == "<<default>>", (
        "guard не сработал: getattr вернул реальный __file__, значит ленивый "
        "импорт выполнился во время обхода sys.modules"
    )
    assert mod.lazy_module is None, "speechbrain.inference импортнулся паразитно"
    assert not [w for w in caught if "deprecated" in str(w.message).lower()]


def test_deep_inspect_stack_does_not_recurse_or_warn() -> None:
    """inspect.stack() из глубокого стека (как в pytorch_lightning) — тихо и без RecursionError."""

    def recurse(depth: int):
        return inspect.stack() if depth == 0 else recurse(depth - 1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frames = recurse(30)

    assert frames
    speechbrain_warnings = [
        str(w.message) for w in caught if "speechbrain" in str(w.message).lower()
    ]
    assert not speechbrain_warnings, (
        f"обход стека спровоцировал импорт speechbrain-модулей: {speechbrain_warnings}"
    )


def test_missing_optional_returns_safe_file_stub() -> None:
    """Ради этого патч и писался: ImportError на optional-пакете не должен течь наружу."""
    checked = 0
    for name in (
        "speechbrain.nnet.loss.transducer_loss",
        "speechbrain.lobes.models.flair",
    ):
        mod = sys.modules.get(name)
        if mod is None or getattr(mod, "lazy_module", "missing") == "missing":
            continue
        checked += 1
        # Либо реальный путь (пакет установлен), либо наша заглушка — но не ImportError.
        assert isinstance(getattr(mod, "__file__", None), str)

        with pytest.raises(AttributeError):
            mod.definitely_not_a_real_symbol_xyz

    if checked == 0:
        pytest.skip("нет зарегистрированных LazyModule для optional-интеграций")
