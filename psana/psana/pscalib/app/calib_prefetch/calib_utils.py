import os
import time
import pickle

GREEN = '\033[92m'
CYAN = '\033[96m'
RESET = '\033[0m'

CALIB_PICKLE_FILENAME = 'calibconst.pkl'

def update_calib(expcode, latest_run, latest_info, log, output_dir, check_before_update=False):
    """
    Update calibration constants by saving latest_info to a pickle file.

    Args:
        expcode (str): Experiment code.
        latest_run (int): Latest run number.
        latest_info (dict): Detector info to store.
        log (Logger): Logger for debug/info output.
        output_dir (str): Output directory for calib pickle.
        check_before_update (bool, optional): If True, check for update necessity. Defaults to False.
    """
    from psana.pscalib.calib.MDBWebUtils import calib_constants_all_types
    fname = os.path.join(output_dir, CALIB_PICKLE_FILENAME)

    if check_before_update and not needs_update(fname, latest_info, output_dir, log):
        log.info(f"{GREEN}[Checked complete]{RESET} - no need to update calib constants.")
        return

    log.info(f"Fetching calib constants for r{latest_run:04d}...")
    try:
        calib_const = {}
        for det_name, det_uid in latest_info.items():
            t0 = time.time()
            calib_const[det_name] = calib_constants_all_types(det_uid, exp=expcode, run=latest_run)
            log.debug(f"Fetched calib_const for {det_name} in {time.time() - t0:.2f}s")
    except Exception as e:
        log.error(f"Failed to create DataSource or retrieve calibconst: {e}")
        return

    filename = CALIB_PICKLE_FILENAME
    temp_path = os.path.join(output_dir, filename + ".inprogress")
    final_path = os.path.join(output_dir, filename)

    t0 = time.time()
    data = {"det_info": latest_info, "calib_const": calib_const}
    with open(temp_path, 'wb') as f:
        pickle.dump(data, f)
    os.rename(temp_path, final_path)
    print(f"{GREEN}[DONE]{RESET} Wrote {final_path} in {time.time() - t0:.2f}s")

def needs_update(fname, latest_info, output_dir, log):
    """
    Determine if the calibration pickle file needs updating.

    Args:
        fname (str): Full path to existing pickle file.
        latest_info (dict): Detector info to compare.
        output_dir (str): Output directory for calib pickle.
        log (Logger): Logger for output.

    Returns:
        bool: True if update needed, False otherwise.
    """
    if not os.path.exists(fname):
        return True

    loaded_data = try_load_data_from_file(output_dir, log)
    saved_info = loaded_data["det_info"]

    if saved_info != latest_info:
        return True

    return False

def try_load_data_from_file(log, output_dir="/dev/shm"):
    """
    Load detector info and calibration constant from the calibration pickle file.

    Args:
        output_dir (str): Directory containing the pickle file.
        log (Logger): Logger for output.

    Returns:
        tuple or None: (run, detector_info) if file exists and loads correctly, else None.
    """
    fname = os.path.join(output_dir, CALIB_PICKLE_FILENAME)
    if not os.path.exists(fname):
        return None

    try:
        with open(fname, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        log.warning(f"Error loading calib pickle file: {e}")
        return None


