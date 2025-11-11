import os
import pickle
import time

from psana.detector.detector_cache import DetectorCacheManager
from psana.pscalib.calib.MDBWebUtils import calib_constants_all_types

GREEN = '\033[92m'
CYAN = '\033[96m'
RESET = '\033[0m'

CALIB_PICKLE_FILENAME = 'calibconst.pkl'

class CalibSource:
    def __init__(self, expcode, run, xtc_dir, output_dir, shmem, detectors, check_before_update, log):
        self.expcode = expcode
        self.run = run
        self.xtc_dir = xtc_dir
        self.output_dir = output_dir
        self.shmem = shmem
        self.check_before_update = check_before_update
        self.log = log
        self.detectors = detectors

    def on_run_begin(self, run):
        """
        Called at the beginning of each run to initialize detectors and prefetch calibration constants.

        Parameters:
        run (psana.Run): The current run object.
        """
        for detname in self.detectors:
            if detname not in run.detnames:
                continue
            det = run.Detector(detname)
            det._run = run  # Attach run to detector for event access
            cache_mgr = DetectorCacheManager(det, check_before_update=self.check_before_update, logger=self.log)
            cache_mgr.ensure()

    def run_loop(self):
        from psana import DataSource
        try:
            if self.shmem:
                ds = DataSource(shmem=self.shmem, skip_calib_load='all', dir=self.xtc_dir)
                self.log.debug(f"Running in shmem mode using id={self.shmem}")
            else:
                ds = DataSource(exp=self.expcode, run=self.run, dir=self.xtc_dir, skip_calib_load='all')
                self.log.debug(f"Running in normal mode using expcode={self.expcode}, run={self.run}")
        except Exception as e:
            self.log.error(f"Failed to create DataSource: {e}")
            raise

        for run in ds.runs():
            expt, runnum, _ = ds._get_runinfo()
            self.log.debug(f"Detected new run: {runnum} {expt=}")
            # Only prefetch calib for detectors present in filtered detinfo
            filtered_detinfo = run.get_filtered_detinfo()
            update_calib(
                latest_run=run,
                latest_info=filtered_detinfo,
                log=self.log,
                output_dir=self.output_dir,
                check_before_update=self.check_before_update,
            )
            # Clear old constants and garbage collect if this isn't the first run
            run._clear_calibconst()
            loaded_data = try_load_data_from_file(self.log, self.output_dir)
            # Attach the retrieved calibconst to be used in Detector caching
            run._calib_const = loaded_data.get('calib_const') or {}
            run.dsparms.calibconst = run._calib_const
            self.on_run_begin(run)

def update_calib(latest_run, latest_info, log, output_dir, check_before_update=False):
    """
    Update calibration constants by saving latest_info to a pickle file.

    Args:
        latest_run (Run): Latest run object.
        latest_info (dict): Detector info to store.
        log (Logger): Logger for debug/info output.
        output_dir (str): Output directory for calib pickle.
        check_before_update (bool, optional): If True, check for update necessity. Defaults to False.
    """
    fname = os.path.join(output_dir, CALIB_PICKLE_FILENAME)

    if check_before_update and not needs_update(fname, latest_info, output_dir, log):
        log.debug(f"{GREEN}[Checked complete]{RESET} - no need to update calib constants.")
        return

    expcode, latest_runnum = latest_run.expt, latest_run.runnum
    log.debug(f"{CYAN}[Updating]{RESET} - {len(latest_info)} detectors in detinfo: {list(latest_info.keys())}")
    log.debug(f"Fetching calib constants for r{latest_runnum:04d}...")
    try:
        calib_const = {}
        for det_name, det_uid in latest_info.items():
            if expcode == "xpptut15":
                det_uid = "cspad_detnum1234"
            t0 = time.time()
            calib_const[det_name] = calib_constants_all_types(det_uid, exp=expcode, run=latest_runnum, dbsuffix="")
            log.debug(f"Fetched calib_const for {det_name} {latest_runnum=} in {time.time() - t0:.2f}s")
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
