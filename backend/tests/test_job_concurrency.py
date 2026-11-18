"""Тесты лимита одновременных тяжёлых джоб (см. runner._heavy_gate).

Зачем это вообще: авто-цепочка download→process на фронте однажды зациклилась и
насоздавала ~20 process-джоб. Каждая честно ушла в `asyncio.to_thread` — и
дефолтный пул executor'а (min(32, cpu+4) = 14 на 10-ядерном Mac) забился 14
инстансами GigaAM, а 15-я и следующие навсегда повисли в очереди executor'а со
статусом running/init/0 % и пустым логом («застряло на Подготовке»).

Фронт починен отдельно, но бэкенд обязан быть устойчив к такому сам: тяжёлая
обработка идёт строго по одной, остальные ЖДУТ В СЕМАФОРЕ и говорят об этом.

Torch тут не нужен: `_run_process_sync` подменяется фейком.
"""

from __future__ import annotations

import asyncio
import threading
import time

from twitch_cut.server import runner as runner_mod
from twitch_cut.server.schemas import ProcessJobRequest
from twitch_cut.server.state import JobKind, JobStatus, JobStore


def _req() -> ProcessJobRequest:
    return ProcessJobRequest(stream="/tmp/stream.mp4")


def test_process_jobs_never_run_in_parallel(monkeypatch, tmp_path):
    """Три джобы стартуют разом — в работе всегда ровно одна, все доезжают."""
    peak = 0
    running = 0
    guard = threading.Lock()

    def fake_sync(req, progress):
        nonlocal peak, running
        with guard:
            running += 1
            peak = max(peak, running)
        time.sleep(0.03)  # даём соседям шанс влезть, если гейта нет
        with guard:
            running -= 1
        return {"decisions_path": str(tmp_path / "decisions.json")}

    monkeypatch.setattr(runner_mod, "_run_process_sync", fake_sync)
    monkeypatch.setattr(runner_mod, "register_project", lambda path: None)

    store = JobStore()
    states = [store.create(JobKind.PROCESS) for _ in range(3)]

    async def go():
        await asyncio.gather(
            *(runner_mod.run_process_job(store, s, _req()) for s in states)
        )

    asyncio.run(go())

    assert peak == 1, f"тяжёлые джобы шли параллельно (пик {peak})"
    assert [s.status for s in states] == [JobStatus.DONE] * 3


def test_queued_process_job_reports_the_queue(monkeypatch, tmp_path):
    """Джоба, ждущая слот, обязана сказать это в message (иначе «0 % навсегда»)."""
    release = threading.Event()

    def fake_sync(req, progress):
        release.wait(timeout=5)
        return {"decisions_path": str(tmp_path / "decisions.json")}

    monkeypatch.setattr(runner_mod, "_run_process_sync", fake_sync)
    monkeypatch.setattr(runner_mod, "register_project", lambda path: None)

    store = JobStore()
    first = store.create(JobKind.PROCESS)
    second = store.create(JobKind.PROCESS)
    messages: list[str] = []

    async def go():
        # Собираем ВСЕ события второй джобы: нам нужен не финальный message, а
        # тот, что она показывала во время ожидания.
        async def collect():
            async for ev in store.subscribe(second.id):
                messages.append(ev["state"]["message"])

        task_first = asyncio.create_task(runner_mod.run_process_job(store, first, _req()))
        await asyncio.sleep(0.05)  # первая успела занять слот
        collector = asyncio.create_task(collect())
        task_second = asyncio.create_task(runner_mod.run_process_job(store, second, _req()))
        await asyncio.sleep(0.05)  # вторая успела упереться в семафор
        assert second.status is JobStatus.RUNNING
        assert "в очереди" in second.message
        release.set()
        await asyncio.gather(task_first, task_second)
        await asyncio.wait_for(collector, timeout=2)

    asyncio.run(go())

    assert any("в очереди" in m for m in messages)
    # После освобождения слота джоба заново объявляет старт — иначе в логе
    # осталась бы «в очереди» на весь прогон.
    assert messages[-1] != "в очереди: уже идёт другая обработка"
    assert second.status is JobStatus.DONE


def test_cancelling_queued_job_frees_the_slot(monkeypatch, tmp_path):
    """Отмена ожидающей джобы: она cancelled, а слот достаётся следующей."""
    release = threading.Event()
    started: list[str] = []

    def fake_sync(req, progress):
        started.append("run")
        release.wait(timeout=5)
        return {"decisions_path": str(tmp_path / "decisions.json")}

    monkeypatch.setattr(runner_mod, "_run_process_sync", fake_sync)
    monkeypatch.setattr(runner_mod, "register_project", lambda path: None)

    store = JobStore()
    first = store.create(JobKind.PROCESS)
    queued = store.create(JobKind.PROCESS)

    async def go():
        task_first = asyncio.create_task(runner_mod.run_process_job(store, first, _req()))
        store.attach_task(first.id, task_first)
        await asyncio.sleep(0.05)
        task_queued = asyncio.create_task(runner_mod.run_process_job(store, queued, _req()))
        store.attach_task(queued.id, task_queued)
        await asyncio.sleep(0.05)

        assert store.cancel(queued.id) is True
        with_cancel = asyncio.gather(task_queued, return_exceptions=True)
        await with_cancel
        assert queued.status is JobStatus.CANCELLED
        assert started == ["run"], "отменённая джоба не должна была стартовать"

        release.set()
        await task_first
        assert first.status is JobStatus.DONE

        # Слот не утёк: следующая джоба проходит.
        third = store.create(JobKind.PROCESS)
        await runner_mod.run_process_job(store, third, _req())
        assert third.status is JobStatus.DONE

    asyncio.run(go())
