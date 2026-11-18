"""Фаза 2: пауза считается ВНУТРИ реплики — регрессия v3 → v3b.

Главный тест здесь — `test_long_turn_with_edge_anchors_only_is_a_pause`. Именно
на этом случае прототип v3 ошибался: контекст брался окном ±25с вокруг реплики,
краевые привязки затягивали ~5 минут текста исходника, и 276-секундный монолог
про свой телефон выглядел как реакция на кадр (в видео тоже говорили про
телефоны). Если этот тест позеленеет при возврате к оконному контексту — значит
он ничего не проверяет.
"""

from __future__ import annotations

from twitch_cut.llm_offtopic.align import Word, align_words, normalize_word
from twitch_cut.llm_offtopic.pause import PAUSE_RATE_PER_S, turn_context, video_context
from twitch_cut.llm_offtopic.turns import Turn


def _words(text: str, *, start: float = 0.0, step: float = 0.5) -> list[Word]:
    out = []
    for i, raw in enumerate(text.split()):
        t = start + i * step
        out.append(Word(text=normalize_word(raw), raw=raw, start_s=t, end_s=t + step / 2))
    return out


# Исходник: 40 слов «истории» про телефоны — тема совпадает с болтовнёй стримера,
# так что уберечь от ложного «по теме» может только пауза.
SOURCE = _words(" ".join(f"слово{i}" for i in range(40)))


def test_playing_video_gives_source_text():
    stream = _words("слово10 слово11 слово12 слово13", start=100.0)
    al = align_words(stream, SOURCE)

    ctx = video_context(al, 99.0, 103.0)

    assert ctx.playing is True
    assert ctx.pause_s == 0.0
    assert ctx.anchors == 1
    assert ctx.text.startswith("слово10 слово11")
    assert "<играет, привязок 1>" in ctx.render()


def test_long_turn_with_edge_anchors_only_is_a_pause():
    """276с монолога, привязки только по краям → ПАУЗА, а не «видео играет».

    Это ровно баг v3: окно ±25с вокруг реплики видело краевые привязки и
    склеивало текст исходника, хотя ВНУТРИ реплики видео молчало.
    """
    stream = (
        # до монолога: видео играет
        _words("слово0 слово1 слово2 слово3", start=0.0)
        # монолог стримера 10.0–286.0 — своих слов в исходнике нет
        + _words(
            " ".join(f"своё{i}" for i in range(60)), start=10.0, step=4.6
        )
        # после монолога стример снял паузу, видео снова звучит
        + _words("слово20 слово21 слово22 слово23", start=290.0)
    )
    al = align_words(stream, SOURCE)
    turn = Turn(start_s=10.0, end_s=286.0, text="…", word_count=60)

    ctx = turn_context(al, turn)

    assert ctx.playing is False, "привязки есть только вне реплики — это пауза"
    assert ctx.anchors == 0
    assert round(ctx.pause_s) == 276
    # Паузу отдаём ЧИСЛОМ: длительность и есть сила улики.
    rendered = ctx.render()
    assert "<ПАУЗА 276с" in rendered
    assert "Перед паузой в видео было" in rendered
    # И это текст ПЕРЕД паузой, а не пять минут склеенного исходника.
    assert "слово3" in rendered
    assert "слово21" not in rendered


def test_video_playing_through_long_turn_is_not_a_pause():
    """Тот же длинный интервал, но видео звучит всё время — паузы нет.

    Обратная сторона теста выше: порог не должен объявлять паузой любую
    длинную реплику.
    """
    stream: list[Word] = []
    for k in range(30):
        # каждые 9с — фраза из исходника (≈0.11 привязки/с, выше порога)
        stream += _words(f"слово{k} слово{k + 1} слово{k + 2} слово{k + 3}", start=k * 9.0)
    al = align_words(stream, SOURCE)

    ctx = video_context(al, 0.0, 270.0)

    assert ctx.playing is True
    assert ctx.rate > PAUSE_RATE_PER_S
    assert ctx.anchors >= 20


def test_pause_before_any_anchor_has_no_source_text():
    # Стример болтает до того, как включил видео: «перед паузой» ещё ничего не было.
    al = align_words(_words("своё1 своё2 своё3 своё4"), SOURCE)
    ctx = video_context(al, 0.0, 100.0)
    assert ctx.playing is False
    assert ctx.text == ""
    assert ctx.render().endswith("—")


def test_single_anchor_in_short_turn_counts_as_playing():
    # Короткая реплика с привязкой: знаменатель не меньше секунды, но одна
    # привязка на 3с — это 0.33 > порога, видео играет.
    stream = _words("слово5 слово6 слово7 слово8", start=50.0)
    ctx = video_context(align_words(stream, SOURCE), 50.0, 53.0)
    assert ctx.playing is True


def test_threshold_is_configurable():
    stream: list[Word] = []
    for k in range(3):
        stream += _words(f"слово{k} слово{k + 1} слово{k + 2} слово{k + 3}", start=k * 30.0)
    al = align_words(stream, SOURCE)
    # 3 привязки на 90с = 0.033/с: ниже дефолтного порога, выше заниженного.
    assert video_context(al, 0.0, 90.0).playing is False
    assert video_context(al, 0.0, 90.0, pause_rate_per_s=0.01).playing is True
