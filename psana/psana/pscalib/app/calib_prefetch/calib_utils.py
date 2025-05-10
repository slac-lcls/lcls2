import os
import time
import pickle
from psana.pscalib.calib.MDBWebUtils import calib_constants_all_types

GREEN = '\033[92m'
CYAN = '\033[96m'
RESET = '\033[0m'

CALIB_PICKLE_FILENAME = 'calibconst.pkl'

class CalibSource:
    def __init__(self, expcode, xtc_dir, output_dir, shmem, check_before_update, log):
        self.expcode = expcode
        self.xtc_dir = xtc_dir
        self.output_dir = output_dir
        self.shmem = shmem
        self.check_before_update = check_before_update
        self.log = log

    def run_loop(self):
        from psana import DataSource
        try:
            if self.shmem:
                ds = DataSource(shmem=self.shmem, skip_calib_load='all', dir=self.xtc_dir)
                self.log.debug(f"Running in shmem mode using id={self.shmem}")
            else:
                ds = DataSource(exp=self.expcode, skip_calib_load='all', dir=self.xtc_dir)
                self.log.debug(f"Running in normal mode using expcode={self.expcode}")
        except Exception as e:
            self.log.error(f"Failed to create DataSource: {e}")
            raise

        for run in ds.runs():
            expt, runnum, _ = ds._get_runinfo()
            self.log.debug(f"Detected new run: {runnum} {expt=}")
            det_info = {k: v.uniqueid for k, v in run.dsparms.configinfo_dict.items()}
            update_calib(
                expcode=expt,
                latest_run=runnum,
                latest_info=det_info,
                log=self.log,
                output_dir=self.output_dir,
                check_before_update=self.check_before_update,
            )

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
    fname = os.path.join(output_dir, CALIB_PICKLE_FILENAME)

    if check_before_update and not needs_update(fname, latest_info, output_dir, log):
        log.debug(f"{GREEN}[Checked complete]{RESET} - no need to update calib constants.")
        return

    log.debug(f"Fetching calib constants for r{latest_run:04d}...")
    try:
        calib_const = {}
        for det_name, det_uid in latest_info.items():
            if expcode == "xpptut15":
                det_uid = "cspad_detnum1234"
            t0 = time.time()
            calib_const[det_name] = calib_constants_all_types(det_uid, exp=expcode, run=latest_run, dbsuffix="")
            log.debug(f"Fetched calib_const for {det_name} {det_uid=} {expcode=} {latest_run=} in {time.time() - t0:.2f}s")
            if not calib_const[det_name]:
                log.warning(f"{det_name} returns {calib_const[det_name]}")
    except Exception as e:
        log.exception(f"Failed to create DataSource or retrieve calibconst: {e}")
        return

    filename = CALIB_PICKLE_FILENAME
    temp_path = os.path.join(output_dir, filename + ".inprogress")
    final_path = os.path.join(output_dir, filename)

    t0 = time.time()
    data = {"det_info": latest_info, "calib_const": calib_const}
    with open(temp_path, 'wb') as f:
        pickle.dump(data, f)
    os.rename(temp_path, final_path)
    log.debug(f"{GREEN}[DONE]{RESET} Wrote {final_path} in {time.time() - t0:.2f}s")

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

    loaded_data = try_load_data_from_file(log, output_dir)
    if not loaded_data:
        return True

    saved_info = loaded_data.get("det_info")
    return saved_info != latest_info

def try_load_data_from_file(log, output_dir="/dev/shm"):
    """
    Load detector info and calibration constant from the calibration pickle file.

    Args:
        log (Logger): Logger for output.
        output_dir (str, optional): Directory containing the pickle file. Defaults to "/dev/shm".

    Returns:
        dict or None: Dictionary containing detector info and calibration constants if successful, else None.
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
