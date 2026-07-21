from subprocess import DEVNULL

import pytest


def test_spawn_console_detaches_xterm_with_devnull(monkeypatch):
    pytest.importorskip("typer")
    import psdaq.slurm.main as main
    from psdaq.slurm.main import Runner

    calls = []

    def fake_popen(args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(main, "Popen", fake_popen)

    runner = object.__new__(Runner)
    result = runner.spawnConsole(
        "bld_0",
        [{"showId": "bld_0", "job_id": "12345"}],
    )

    assert result == 0
    assert calls == [
        (
            [Runner.PATH_XTERM, "-T", "bld_0", "-e", "sattach 12345.0"],
            {
                "stdin": DEVNULL,
                "stdout": DEVNULL,
                "stderr": DEVNULL,
                "start_new_session": True,
            },
        )
    ]
