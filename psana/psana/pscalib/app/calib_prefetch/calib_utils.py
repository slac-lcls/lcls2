import os
import time
import pickle
from psana.utils import Logger

GREEN = '\033[92m'
CYAN = '\033[96m'
RESET = '\033[0m'

def get_latest_run(output_dir, expcode, log, xtc_dir=None):
    """
    Return the latest usable run number based on xtc2 files.

    Args:
        output_dir (str): Directory where calibconst files are written.
        expcode (str): Experiment code.
        log (Logger): Logger object for debug/info output.
        xtc_dir (str, optional): Directory containing xtc2 files.

    Returns:
        int or None: Latest usable run number or None if not found.
    """
    """Return the latest usable run number for an experiment based on available xtc2 files."""
    from psana import DataSource
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
        existing_file = os.path.join(output_dir, f"calibconst_{expcode}_r{runnum:04d}.pkl")
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
    """
    Return the highest run number with a stored calibration pickle file.

    Args:
        output_dir (str): Directory where calibration pickle files are stored.
        expcode (str): Experiment code.

    Returns:
        int: Highest run number found or -1 if none exists.
    """
    """Return the highest run number for which a calibration pickle file exists."""
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
    """
    Extract detector unique IDs from a DataSource run.

    Args:
        expcode (str): Experiment code.
        runnum (int): Run number.
        xtc_dir (str, optional): Directory to locate xtc2 files.

    Returns:
        dict: Mapping of detector names to unique IDs.
    """
    """Extract detector unique IDs from the configinfo of a given run."""
    # TODO: Handle different run types. The DataSource initiation doesn't work
    # with RunShmem or RunSingleFile.
    from psana import DataSource
    try:
        ds = DataSource(exp=expcode, run=runnum, skip_calib_load='all', dir=xtc_dir)
    except Exception as e:
        Logger().debug(f'Exception failed to create DataSource: {e}')
        return {}
    configinfo_dict = ds.dsparms.configinfo_dict
    return {det_name: cfg.uniqueid for det_name, cfg in configinfo_dict.items()}

def detector_info_from_file(output_dir, expcode, runnum):
    """
    Load stored detector info dictionary from calibration file.

    Args:
        output_dir (str): Directory containing calibration pickle files.
        expcode (str): Experiment code.
        runnum (int): Run number.

    Returns:
        dict or None: Detector info or None if load fails.
    """
    """Load saved detector info dictionary from a calibration pickle file."""
    path = os.path.join(output_dir, f"calibconst_{expcode}_r{runnum:04d}.pkl")
    try:
        with open(path, 'rb') as f:
            det_info = pickle.load(f)["det_info"]
            if not isinstance(det_info, dict):
                raise ValueError(f"Invalid det_info structure in {path}")
        return det_info
    except Exception as e:
        Logger().debug(f'Exception failed to get det_info from {path}: {e}')
        return None

def needs_update(latest_info, existing_info):
    """
    Check if calibration data needs update based on detector info mismatch.

    Args:
        latest_info (dict): Detector info for latest run.
        existing_info (dict): Stored detector info from previous run.

    Returns:
        bool: True if update is needed, False otherwise.
    """
    """Return True if the detector info has changed or is incomplete."""
    return (
        not set(latest_info.keys()).issubset(existing_info.keys()) or
        any(latest_info[k] != existing_info.get(k) for k in latest_info)
    )

def try_load_calib_const_from_file(output_dir, expcode):
    """
    Load calibration constants from latest existing pickle file.

    Args:
        output_dir (str): Directory with calibration pickle files.
        expcode (str): Experiment code.

    Returns:
        tuple: (calib_const dict, run number)
    """
    """Attempt to load calibration constants from an existing pickle file."""
    run = load_existing_run(output_dir, expcode)
    if run == -1:
        return {}, -1
    path = os.path.join(output_dir, f"calibconst_{expcode}_r{run:04d}.pkl")
    try:
        with open(path, 'rb') as f:
            calib_const = pickle.load(f)["calib_const"]
        return calib_const, run
    except Exception as e:
        print(f"[WARN] Failed to load calib_const from {path}: {e}")
        return {}, -1

def ensure_valid_calibconst(expcode, latest_run, latest_info, xtc_dir, skip_calib_load, output_dir='/dev/shm', log=None):
    """
    Ensure calibration constants for given run are valid, updating if needed.

    Args:
        expcode (str): Experiment code.
        latest_run (int): Run number.
        latest_info (dict): Detector info for latest run.
        xtc_dir (str): Directory containing xtc2 files.
        skip_calib_load (list): List of detectors to skip.
        output_dir (str): Path to write/read calibration pickle files.
        log (Logger, optional): Logger for logging.

    Returns:
        tuple: (calib_const dict, run number)
    """
    """Ensure calibration constants are available and up-to-date for a given run."""
    if log is None:
        log = Logger()
    existing_run = load_existing_run(output_dir, expcode)
    log.debug(f"ensure_valid_calibconst for {latest_run=} ({existing_run=})")
    try:
        existing_info = detector_info_from_file(output_dir, expcode, existing_run)
        if needs_update(latest_info, existing_info):
            log.debug("Existing calibconst is outdated, updating...")
            update_calib(expcode, latest_run, latest_info=latest_info, output_dir=output_dir, log=log, xtc_dir=xtc_dir, skip_calib_load=skip_calib_load)
        else:
            log.debug("Existing calibconst is up-to-date.")
    except Exception as e:
        log.warning(f"Failed to validate existing calibconst: {e}")
        update_calib(expcode, latest_run, latest_info=latest_info, output_dir=output_dir, log=log, xtc_dir=xtc_dir, skip_calib_load=skip_calib_load)

    return try_load_calib_const_from_file(output_dir, expcode)

def update_calib(expcode, latest_run=None, latest_info=None, output_dir='/dev/shm', log=None, xtc_dir=None, skip_calib_load=[]):
    """
    Fetch calibration constants and save them if detector info changed.

    Args:
        expcode (str): Experiment code.
        latest_run (int, optional): Run number to use.
        latest_info (dict, optional): Detector info to use for comparison.
        output_dir (str): Directory to store calibration constants.
        log (Logger, optional): Logger instance.
        xtc_dir (str, optional): Directory for xtc2 files.
        skip_calib_load (list): Detectors to skip loading.

    Returns:
        None
    """
    """Download and store calibration constants to a pickle file if update is needed."""
    if log is None:
        log = Logger()

    from psana.pscalib.calib.MDBWebUtils import calib_constants_all_types

    log.debug(f"{CYAN}[START]{RESET} Checking calibration for exp={expcode}")
    if latest_run is None:
        latest_run = get_latest_run(output_dir, expcode, log, xtc_dir=xtc_dir)
        log.debug(f"Found latest_run={latest_run}")
    else:
        log.debug(f" {latest_run=}")

    if latest_run is None:
        return

    existing_run = load_existing_run(output_dir, expcode)
    log.debug(f"Check existing_run={existing_run}")

    try:
        if latest_info is None:
            latest_info = detector_info_from_run(expcode, latest_run, xtc_dir)
        existing_info = detector_info_from_file(output_dir, expcode, existing_run) if existing_run >= 0 else {}

        log.debug(f"latest_info keys={latest_info.keys()}")
        log.debug(f"existing_info keys={existing_info.keys()}")

        if not needs_update(latest_info, existing_info):
            log.debug(f"Detector info unchanged from r{existing_run:04d} to r{latest_run:04d}, skipping.")
            return

    except Exception as e:
        log.error(f"Failed to compare detector info: {e}")
        return

    log.info(f"Fetching calib constants for r{latest_run:04d}...")
    try:
        calib_const = {}
        for det_name, det_uid in latest_info.items():
            t0 = time.time()
            # For our unittest, the det_uid is not defined in Configure.
            if expcode == "xpptut15":
                det_uid = "cspad_detnum1234"
                log.warning(f"UNITTEST define {det_uid=}")
            if det_name in skip_calib_load:
                calib_const[det_name] = None
                continue
            calib_const[det_name] = calib_constants_all_types(det_uid, exp=expcode, run=latest_run)
            print(f"Fetched calib_const for {det_name} in {time.time() - t0:.2f}s")
    except Exception as e:
        log.error(f"Failed to create DataSource or retrieve calibconst: {e}")
        return

    filename = f"calibconst_{expcode}_r{latest_run:04d}.pkl"
    temp_path = os.path.join(output_dir, filename + ".inprogress")
    final_path = os.path.join(output_dir, filename)

    t0 = time.time()
    data = {"det_info": latest_info, "calib_const": calib_const}
    with open(temp_path, 'wb') as f:
        pickle.dump(data, f)
    os.rename(temp_path, final_path)
    print(f"{GREEN}[DONE]{RESET} Wrote {final_path} in {time.time() - t0:.2f}s")
