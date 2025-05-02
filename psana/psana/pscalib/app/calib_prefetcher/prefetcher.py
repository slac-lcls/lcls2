import os
import pickle
import time

import zstandard as zstd

from psana import DataSource
from psana.pscalib.calib.MDBWebUtils import calib_constants_all_types
from psana.utils import Logger

# ANSI escape codes for colorr
GREEN = '\033[92m'
CYAN = '\033[96m'
RESET = '\033[0m'


def get_det_uniqueid(expcode, runnum, det_name):
    ds = DataSource(exp=expcode, run=runnum, dir='/cds/data/drpsrcf/mfx/mfx101332224/xtc')
    configinfo_dict = ds.dsparms.configinfo_dict
    if det_name not in configinfo_dict:
        raise ValueError(f"Detector name '{det_name}' not found in configinfo_dict.")
    return configinfo_dict[det_name].uniqueid


def fetch_and_store(expcode, runnum, det_name, dbsuffix='', output_dir='/dev/shm', log=None, compress=True):
    if log is None:
        log = Logger()

    t0 = time.time()
    log.info(f"{CYAN}[START]{RESET} Fetching calibration for {det_name} (exp={expcode}, run={runnum})")

    det_uid = get_det_uniqueid(expcode, runnum, det_name)
    log.debug(f"Resolved det_uniqueid: {det_uid}")

    t1 = time.time()
    calib_const = calib_constants_all_types(det_uid, exp=expcode, run=runnum, dbsuffix=dbsuffix)
    log.debug(f"Fetched calib constants in {time.time() - t1:.2f}s")

    t2 = time.time()
    pkl_data = pickle.dumps(calib_const)
    log.debug(f"Serialized in {time.time() - t2:.2f}s")

    if compress:
        t3 = time.time()
        cctx = zstd.ZstdCompressor(level=3)
        blob = cctx.compress(pkl_data)
        log.debug(f"Compressed in {time.time() - t3:.2f}s")
        ext = ".zst"
    else:
        blob = pkl_data
        ext = ".pkl"

    filename = f"calib_const-{expcode}-{runnum}{ext}"
    temp_filename = filename + ".inprogress"
    temp_path = os.path.join(output_dir, temp_filename)
    final_path = os.path.join(output_dir, filename)

    t4 = time.time()
    with open(temp_path, 'wb') as f:
        f.write(blob)
    os.rename(temp_path, final_path)
    log.info(f"Wrote calibration to {final_path} ({len(blob)/1e6:.2f} MB) in {time.time() - t4:.2f}s")

    elapsed = time.time() - t0
    log.info(f"{GREEN}[DONE]{RESET} Fetch complete in {elapsed:.2f}s. Sleeping until next interval...")

