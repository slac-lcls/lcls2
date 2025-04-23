import os
import tempfile
import shutil
import logging
import pytest

from psdaq.slurm.utils import run_slurm_with_retries  # Replace with the actual module path

logger = logging.getLogger(__name__)


@pytest.fixture
def fake_sbatch_script():
    """
    Creates a fake sbatch script that fails first two times, then succeeds.
    """
    tmp_dir = tempfile.mkdtemp()
    script_path = os.path.join(tmp_dir, "sbatch")

    # This file keeps track of how many times the script was called
    count_file = os.path.join(tmp_dir, "count.txt")

    script_content = f"""#!/bin/bash
COUNT=$(cat "{count_file}" 2>/dev/null || echo 0)
COUNT=$((COUNT + 1))
echo $COUNT > "{count_file}"

if [ "$COUNT" -lt 3 ]; then
    echo "Simulated failure" >&2
    exit 1
else
    echo "Simulated success"
    exit 0
fi
"""
    with open(script_path, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)

    yield tmp_dir

    # Cleanup
    shutil.rmtree(tmp_dir)


def test_run_slurm_with_retries_retries_successfully(fake_sbatch_script, caplog):
    """
    Tests that run_slurm_with_retries retries on failure and eventually succeeds.
    """
    fake_path = fake_sbatch_script
    old_path = os.environ["PATH"]
    os.environ["PATH"] = f"{fake_path}:{old_path}"

    caplog.set_level(logging.WARNING)

    output = run_slurm_with_retries("sbatch", max_retries=5, retry_delay=1)

    assert "Simulated success" in output
    assert "Retrying in" in caplog.text
    assert "Attempt 2/5" in caplog.text
    assert "Attempt 3/5" not in caplog.text  # It should succeed on the 3rd try

    os.environ["PATH"] = old_path
