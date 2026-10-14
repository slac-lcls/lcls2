"""Bounded multi-rank regression for rank-local geometry cache misses."""
import os
from pathlib import Path
import subprocess
import sys


def test_geometry_cache_mpi():
    env = dict(os.environ, PS_PARALLEL='mpi', PS_GEO_SHARE='1')
    subprocess.run(
        ['mpirun', '-n', '3', sys.executable,
         str(Path(__file__).with_name('mpi_geometry_cache.py'))],
        env=env, check=True, timeout=90,
    )
