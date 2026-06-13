import sys

import pytest

pytest.importorskip("PyQt5")

from psdaq.slurm import daqstat


@pytest.mark.parametrize("option", ["-h", "-v", "--help", "--version"])
def test_help_options_print_usage_without_argument(monkeypatch, capsys, option):
    monkeypatch.setattr(sys, "argv", ["daqstat", option])

    assert daqstat.main() == 0

    output = capsys.readouterr()
    assert "option -h requires argument" not in output.out
    assert "option -v requires argument" not in output.out
    assert "Usage:" in output.out
