import importlib

import pytest

MODULES = ["cnftopy", "config", "gen_gres_conf", "main", "subproc", "utils"]


def test_psdaq_slurm_warns():
    import psdaq.slurm

    with pytest.warns(DeprecationWarning, match="psalg.slurm"):
        importlib.reload(psdaq.slurm)


@pytest.mark.parametrize("name", MODULES)
def test_psdaq_slurm_aliases_psalg_slurm(name):
    old = importlib.import_module(f"psdaq.slurm.{name}")
    new = importlib.import_module(f"psalg.slurm.{name}")
    assert old is new
