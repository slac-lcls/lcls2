import os
import subprocess
import sys


def test_serial_import_does_not_load_mpi():
    """A serial reader subprocess must not initialize MPI during import."""
    env = os.environ.copy()
    env["PS_PARALLEL"] = "none"
    code = """
import sys
import psana
import psana.detector.UtilsJungfrau

assert "mpi4py.MPI" not in sys.modules, "Serial psana import loaded MPI"
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
