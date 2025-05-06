import argparse
import logging
import signal
import sys

from psana import DataSource
from psana.pscalib.app.calib_prefetch import calib_utils
from psana.utils import Logger

def main():
    """
    Entry point for the calibration prefetcher CLI tool.

    Parses command-line arguments, sets up logging and signal handlers,
    and periodically calls `update_calib()` to fetch calibration constants
    for the latest available run of an experiment or via shared memory.
    """
    parser = argparse.ArgumentParser(description="Calibration Prefetcher")
    parser.add_argument('-e', '--expcode', default=None, help='Experiment code')
    parser.add_argument('--xtc-dir', default=None, help='Optional path to XTC directory')
    parser.add_argument('--output-dir', default='/dev/shm', help='Output directory')
    parser.add_argument('--log-level', default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument('--timestamp', action='store_true', help='Include timestamp in log messages')
    parser.add_argument('--shmem', default=None, help='Shmem ID to run in shared memory mode')
    parser.add_argument('--check-before-update', default=False, help='Force check before we update the calibration constants')
    args = parser.parse_args()

    log = Logger(timestamp=args.timestamp)
    log.logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    def signal_handler(sig, frame):
        log.info("Shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        if args.shmem:
            log.info(f"Running in shmem mode using id={args.shmem}")
            ds = DataSource(shmem=args.shmem, skip_calib_load='all', dir=args.xtc_dir)
        else:
            log.info(f"Running in normal mode using expcode={args.expcode}")
            ds = DataSource(exp=args.expcode, skip_calib_load='all', dir=args.xtc_dir)
    except Exception as e:
            log.error(f"Failed to create DataSource: {e}")
            raise

    for run in ds.runs():
        runnum = run.runnum
        log.info(f"Detected new run: {runnum}")
        det_info = run.dsparms.configinfo_dict
        det_uid_map = {k: v.uniqueid for k, v in det_info.items()}
        calib_utils.update_calib(
            args.expcode,
            latest_run=runnum,
            latest_info=det_uid_map,
            log=log,
            output_dir=args.output_dir,
            check_before_update=args.check_before_update,
        )

if __name__ == '__main__':
    main()
