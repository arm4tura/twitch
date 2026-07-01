import pytest

from twitch_cut.vegas_export import VegasExportError, csharp_string, exportable_mutes, generate_vegas_script


def test_csharp_string_escapes_label():
    assert csharp_string(r'a "quote" \ path') == r'"a \"quote\" \\ path"'


def test_generate_vegas_script_splits_and_zero_gains_and_groups():
    decisions = {
        "mutes": [
            {
                "stream_start": 64.22,
                "stream_end": 64.82,
                "word": "блин",
                "intro_risk": True,
            }
        ]
    }
    script = generate_vegas_script(decisions)
    # header + Vegas namespace
    assert "using ScriptPortal.Vegas;" in script
    # no regions/markers at all
    assert "AddRegion" not in script
    assert "Regions.Add" not in script
    assert "Marker" not in script
    # per-mute call is the silence helper, with formatted seconds
    assert "SilenceAudioRange(vegas, 64.220, 64.820);" in script
    # splitting is preserved (это и есть \"нарезать момент\")
    assert "Split(Timecode.FromSeconds" in script
    # ключевое: gain=0, а не Mute=true. Проверяем что нет активного statement'а
    # вида `<что-то>.Mute = true;` (комментарии с этим текстом допустимы —
    # контр-пример в докстринге).
    assert "audioEvent.Gain = 0f;" in script
    assert "target.Mute = true;" not in script
    for line in script.splitlines():
        code = line.split("//", 1)[0]
        assert ".Mute = true" not in code, f"unexpected Mute=true in code line: {line!r}"
    # группировка треков после всех разрезов
    assert "GroupAllTracks(vegas);" in script
    assert "new TrackGroup()" in script
    assert "vegas.Project.Groups.Add(group);" in script
    assert "track.Group = group;" in script


def test_generate_vegas_script_calls_group_even_with_no_mutes():
    # Пустой список мутов — всё равно грузим GroupAllTracks, иначе после
    # ре-запуска скрипта на уже-обработанном проекте группировка не восстановится.
    script = generate_vegas_script({"mutes": []})
    # Определение хелпера в скрипте есть всегда; важно, что нет ВЫЗОВОВ.
    assert "SilenceAudioRange(vegas," not in script
    assert "GroupAllTracks(vegas);" in script


def test_exportable_mutes_skips_rejected_and_non_mute():
    decisions = {
        "mutes": [
            {"stream_start": 1.0, "stream_end": 1.5, "word": "keep", "action": "MUTE"},
            {"stream_start": 2.0, "stream_end": 2.5, "word": "rejected", "status": "rejected"},
            {"stream_start": 3.0, "stream_end": 3.5, "word": "keep-action", "action": "KEEP"},
        ]
    }
    exported = exportable_mutes(decisions)
    assert len(exported) == 1
    assert exported[0]["word"] == "keep"

    script = generate_vegas_script(decisions)
    # только "keep" получает splice — остальные два не должны генерировать вызовы
    assert script.count("SilenceAudioRange(vegas,") == 1
    assert "SilenceAudioRange(vegas, 1.000, 1.500);" in script
    assert "SilenceAudioRange(vegas, 2.000" not in script
    assert "SilenceAudioRange(vegas, 3.000" not in script


def test_exportable_mutes_validates_required_times():
    with pytest.raises(VegasExportError, match="mutes"):
        exportable_mutes({})
    with pytest.raises(VegasExportError, match="stream_start"):
        exportable_mutes({"mutes": [{"stream_end": 1.0}]})
    with pytest.raises(VegasExportError, match="stream_end <= stream_start"):
        exportable_mutes({"mutes": [{"stream_start": 2.0, "stream_end": 1.0}]})
