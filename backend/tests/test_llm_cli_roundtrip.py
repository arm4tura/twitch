"""End-to-end CLI test: highlights-export → подделываем JSON-ответ →
highlights-import → в decisions.json лежит ключ 'highlights'."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from twitch_cut.cli import app

runner = CliRunner()


def _mk_transcript(n_words: int) -> dict:
    return {
        "segments": [
            {
                "start": 0.0,
                "end": float(n_words),
                "words": [
                    {"word": f"w{i}", "start": float(i), "end": i + 0.5}
                    for i in range(n_words)
                ],
            }
        ]
    }


def _prep_workspace(tmp_path: Path) -> tuple[Path, Path]:
    transcript = tmp_path / "cache" / "transcript.json"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text(
        json.dumps(_mk_transcript(120), ensure_ascii=False), encoding="utf-8"
    )
    decisions = tmp_path / "decisions.json"
    decisions.write_text(
        json.dumps(
            {
                "mutes": [],
                "cuts": [],
                "caches": {"transcript": str(transcript)},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return decisions, transcript


def test_highlights_export_creates_package(tmp_path):
    decisions, _ = _prep_workspace(tmp_path)
    out = tmp_path / "notebooklm"

    result = runner.invoke(
        app,
        [
            "highlights-export",
            "--decisions", str(decisions),
            "--out-dir", str(out),
            "--n-highlights", "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "prompt.md").exists()
    assert (out / "transcript_001.md").exists()
    assert (out / "manifest.json").exists()


def test_highlights_import_roundtrip(tmp_path):
    decisions, _ = _prep_workspace(tmp_path)
    out = tmp_path / "notebooklm"

    runner.invoke(
        app,
        [
            "highlights-export",
            "--decisions", str(decisions),
            "--out-dir", str(out),
            "--n-highlights", "2",
        ],
    )

    # Подделываем ответ NotebookLM — валидный, с двумя highlights внутри
    # диапазона транскрипта [0..120].
    response = tmp_path / "response.json"
    response.write_text(
        "```json\n"
        + json.dumps(
            {
                "highlights": [
                    {
                        "start_s": 5.0,
                        "end_s": 45.0,
                        "title": "первый",
                        "reason": "тест",
                        "score": 0.9,
                    },
                    {
                        "start_s": 60.0,
                        "end_s": 105.0,
                        "title": "второй",
                        "reason": "тест 2",
                        "score": 0.8,
                    },
                ]
            }
        )
        + "\n```",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "highlights-import",
            "--decisions", str(decisions),
            "--response", str(response),
            "--output", str(decisions),
        ],
    )
    assert result.exit_code == 0, result.output

    updated = json.loads(decisions.read_text(encoding="utf-8"))
    assert "highlights" in updated
    assert len(updated["highlights"]["highlights"]) == 2
    assert updated["highlights"]["highlights"][0]["title"] == "первый"
    # transcript_hash проставлен, т.к. транскрипт доступен.
    assert updated["highlights"]["transcript_hash"] is not None


def test_highlights_import_reports_validation_errors(tmp_path):
    decisions, _ = _prep_workspace(tmp_path)
    response = tmp_path / "response.json"
    # end < start — importer должен упасть с человекочитаемой ошибкой.
    response.write_text(
        json.dumps(
            {
                "highlights": [
                    {
                        "start_s": 100.0,
                        "end_s": 50.0,
                        "title": "плохой",
                        "reason": "инвертирован",
                        "score": 0.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "highlights-import",
            "--decisions", str(decisions),
            "--response", str(response),
            "--output", str(decisions),
        ],
    )
    assert result.exit_code == 1
    assert "end_s" in result.output
