import os
import pickle
import time

from psana import DataSource
from psana.utils import Logger

GREEN = '\033[92m'
CYAN = '\033[96m'
RESET = '\033[0m'

def get_latest_run(expcode, log, xtc_dir=None):
    if not os.path.exists(xtc_dir):
        log.warning(f"No such directory: {xtc_dir}")
        return None

    usable_runs = []
    for fname in os.listdir(xtc_dir):
        if not fname.endswith(".xtc2"):
            continue
        parts = fname.split('-')
        if len(parts) < 2 or not parts[1].startswith('r'):
            continue
        runnum = int(parts[1][1:])
        if runnum not in usable_runs:
            usable_runs.append(runnum)

    if not usable_runs:
        log.info("No smalldata files found yet or no unique run numbers.")
        return None

    for runnum in sorted(usable_runs, reverse=True):
        existing_file = os.path.join("/dev/shm", f"calibconst_{expcode}_r{runnum:04d}.pkl")
        if os.path.exists(existing_file):
            log.info(f"Run r{runnum:04d} already processed.")
            return None
        try:
            ds = DataSource(exp=expcode, run=runnum, dir=xtc_dir, skip_calib_load='all')
            _ = next(ds.runs())
            log.debug(f"Run r{runnum:04d} is usable.")
            return runnum
        except Exception as e:
            log.warning(f"Run r{runnum:04d} found in files but is not usable: {e}")

    log.info("No usable runs found.")
    return None

def load_existing_run(output_dir, expcode):
    highest = -1
    for fname in os.listdir(output_dir):
        if not fname.startswith(f"calibconst_{expcode}_r") or not fname.endswith(".pkl"):
            continue
        try:
            runnum = int(fname.split('_r')[-1].split('.')[0])
            highest = max(highest, runnum)
        except ValueError:
            continue
    return highest

def detector_info_from_run(expcode, runnum, xtc_dir=None):
    ds = DataSource(exp=expcode, run=runnum, skip_calib_load='all', dir=xtc_dir)
    configinfo_dict = ds.dsparms.configinfo_dict
    return {det_name: cfg.uniqueid for det_name, cfg in configinfo_dict.items()}

def needs_update(latest_info, existing_info):
    return (
        not set(latest_info.keys()).issubset(existing_info.keys()) or
        any(latest_info[k] != existing_info.get(k) for k in latest_info)
    )

def try_load_calib_const_from_file(output_dir, expcode):
    run = load_existing_run(output_dir, expcode)
    if run == -1:
        return None, -1
    path = os.path.join(output_dir, f"calibconst_{expcode}_r{run:04d}.pkl")
    try:
        with open(path, 'rb') as f:
            calib_const = pickle.load(f)
        return calib_const, run
    except Exception:
        return None, -1

def update_calib(expcode, output_dir='/dev/shm', log=None, xtc_dir=None):
    if log is None:
        log = Logger()

    log.info(f"{CYAN}[START]{RESET} Checking calibration for exp={expcode}")
    latest_run = get_latest_run(expcode, log, xtc_dir=xtc_dir)
    log.info(f"Found latest_run={latest_run}")
    if latest_run is None:
        return

    existing_run = load_existing_run(output_dir, expcode)
    log.debug(f"Check existing_run={existing_run}")

    try:
        latest_info = detector_info_from_run(expcode, latest_run, xtc_dir)
        existing_info = detector_info_from_run(expcode, existing_run, xtc_dir) if existing_run >= 0 else {}

        log.debug(f"latest_info keys={latest_info.keys()}")
        log.debug(f"existing_info keys={existing_info.keys()}")

        if not needs_update(latest_info, existing_info):
            log.info(f"Detector info unchanged from r{existing_run:04d} to r{latest_run:04d}, skipping.")
            return

    except Exception as e:
        log.error(f"Failed to compare detector info: {e}")
        return

    t_fetch = time.time()
    log.info(f"Fetching calib constants for r{latest_run:04d}...")
    try:
        ds = DataSource(exp=expcode, run=latest_run, dir=xtc_dir)
        next(ds.runs())
        calib_const = ds.dsparms.calibconst
        log.info(f"Fetched calib_const in {time.time() - t_fetch:.2f}s")
    except Exception as e:
        log.error(f"Failed to create DataSource or retrieve calibconst: {e}")
        return

    filename = f"calibconst_{expcode}_r{latest_run:04d}.pkl"
    temp_path = os.path.join(output_dir, filename + ".inprogress")
    final_path = os.path.join(output_dir, filename)

    t0 = time.time()
    with open(temp_path, 'wb') as f:
        pickle.dump(calib_const, f)
    os.rename(temp_path, final_path)
    log.info(f"{GREEN}[DONE]{RESET} Wrote {final_path} in {time.time() - t0:.2f}s")
