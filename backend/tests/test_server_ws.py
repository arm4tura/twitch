"""WS-тест: подключение к джобе и получение стрима событий вплоть до final.

Используем TestClient.websocket_connect. Джобу симулируем через быстрый
highlights-export на маленьком транскрипте — реальный runner, реальная очередь.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from twitch_cut.server.app import create_app


def _mk_transcript(n_words: int) -> dict:
    return {
        "segments": [
            {
                "start": 0.0, "end": float(n_words),
                "words": [{"word": f"w{i}", "start": float(i), "end": i + 0.5} for i in range(n_words)],
            }
        ]
    }


def test_ws_stream_delivers_final(tmp_path: Path):
    # ВАЖНО: TestClient оборачиваем в `with`. Starlette `_portal_factory`
    # создаёт новый anyio blocking_portal на КАЖДЫЙ вызов, если `self.portal`
    # не установлен (а он ставится только в `__enter__`). Portal closing вызывает
    # `_cancel_all_tasks(loop)` — и наша фоновая asyncio.Task с runner'ом
    # получает CancelledError между `await asyncio.to_thread(...)` и
    # `_finalize(DONE)`. Итог: final приходит со status=cancelled ~40% раз.
    # Один shared portal на весь тест — POST и websocket_connect делят loop.
    app = create_app()

    tpath = tmp_path / "t.json"
    tpath.write_text(json.dumps(_mk_transcript(30)), encoding="utf-8")
    dpath = tmp_path / "d.json"
    dpath.write_text(json.dumps({"caches": {"transcript": str(tpath)}}), encoding="utf-8")

    with TestClient(app) as client:
        resp = client.post(
            "/jobs/highlights-export",
            json={
                "decisions": str(dpath),
                "out_dir": str(tmp_path / "pkg"),
                "n_highlights": 2,
            },
        )
        assert resp.status_code == 200
        job_id = resp.json()["id"]

        events: list[dict] = []
        with client.websocket_connect(f"/jobs/{job_id}/events") as ws:
            # Читаем пока не увидим final; TestClient WS блокируется, но задача
            # уже стартовала в фоне, финал прилетит быстро.
            while True:
                try:
                    msg = ws.receive_text()
                except Exception:
                    break
                data = json.loads(msg)
                events.append(data)
                if data.get("type") == "final":
                    break

    assert events, "no events received"
    final = events[-1]
    assert final["type"] == "final"
    assert final["state"]["status"] == "done"
    assert final["state"]["result"]["out_dir"] == str(tmp_path / "pkg")


def test_ws_unknown_job_closes_with_error(tmp_path: Path):
    app = create_app()
    with TestClient(app) as client:
        with client.websocket_connect("/jobs/nope/events") as ws:
            msg = ws.receive_text()
            data = json.loads(msg)
            assert data["type"] == "error"
