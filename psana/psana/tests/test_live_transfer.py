import logging
import shutil
import subprocess
from pathlib import Path

import pytest

from psana import DataSource


def _resolve_binary(name):
    exe_path = shutil.which(name)
    if exe_path:
        return Path(exe_path)

    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "install" / "bin" / name,
        repo_root / "xtcdata" / "build" / "xtcdata" / "app" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


SMDWRITER = _resolve_binary("smdwriter")
XTCWRITER = _resolve_binary("xtcwriter")


@pytest.mark.slow
@pytest.mark.skipif(
    SMDWRITER is None,
    reason="smdwriter executable is required for the live transfer test",
)
@pytest.mark.skipif(
    XTCWRITER is None,
    reason="xtcwriter executable is required for the live transfer test",
)
def test_live_mode_detects_completed_transfers(tmp_path, caplog, monkeypatch):
    """Ensure SmdReaderManager notices .inprogress → .xtc2 flips."""
    exp = "codexp9999"
    run = 175
    xtc_dir = tmp_path / "xtc"
    smd_dir = xtc_dir / "smalldata"

    # Step 1: create tmp folder & generate a bigdata file with the expected
    # naming pattern.
    xtc_dir.mkdir()
    xtc_file = xtc_dir / f"{exp}-r{run:04d}-s000-c000.xtc2"
    subprocess.check_call(
        [
            str(XTCWRITER),
            "-f",
            str(xtc_file),
            "-n",
            "200",
        ]
    )

    # Step 2: create the smalldata folder inside the tmp dir.
    smd_dir.mkdir()
    smd_inprogress = smd_dir / f"{exp}-r{run:04d}-s000-c000.smd.xtc2.inprogress"
    smd_final = smd_dir / f"{exp}-r{run:04d}-s000-c000.smd.xtc2"
    if smd_inprogress.exists():
        smd_inprogress.unlink()
    if smd_final.exists():
        smd_final.unlink()

    # Step 3: run smdwriter with -m 10 to throttle writes, producing an
    # .inprogress file.
    writer_cmd = [
        str(SMDWRITER),
        "-f",
        str(xtc_file),
        "-o",
        str(smd_inprogress),
        "-m",
        "10",
        "-n",
        "200",
    ]
    subprocess.check_call(writer_cmd, cwd=xtc_dir)
    assert smd_inprogress.exists()
    assert smd_inprogress.stat().st_size > 0

    monkeypatch.setenv("PS_R_MAX_RETRIES", "30")
    caplog.clear()
    caplog.set_level(logging.DEBUG)

    # Avoid hitting the live logbook database — force psana to trust on-disk files.
    monkeypatch.setattr(
        "psana.psexp.ds_base.DataSourceBase._get_file_info_from_db",
        lambda self, runnum: {},
    )

    # Step 4: run a live-mode DataSource while writing is ongoing.
    events_seen = 0
    transfer_flipped = False
    try:
        ds = DataSource(
            exp=exp,
            run=run,
            dir=str(xtc_dir),
            live=True,
            log_level="DEBUG",
            batch_size=1,
            max_events=5,
        )
        run_obj = next(ds.runs())
        for evt in run_obj.events():
            events_seen += 1
            if not transfer_flipped and smd_inprogress.exists():
                # Step 3b: mimic the `mv` command once psana has started reading.
                smd_inprogress.rename(smd_final)
                transfer_flipped = True
            if events_seen >= 3:
                break
    finally:
        if not transfer_flipped and smd_inprogress.exists():
            smd_inprogress.rename(smd_final)
        transfer_flipped = True

    assert events_seen > 0, (
        "DataSource failed to read any events while the file was live"
    )
    assert smd_final.exists(), "final .xtc2 file was never created"
    assert not smd_inprogress.exists(), (
        ".inprogress file should have been flipped to .xtc2"
    )

    transfer_flag = run_obj.smdr_man.check_transfer_complete()
    transfer_logged = any(
        "All .inprogress files finalized" in record.message for record in caplog.records
    )
    assert transfer_flag or transfer_logged, (
        "SmdReaderManager never detected that .inprogress files finished transferring"
    )
