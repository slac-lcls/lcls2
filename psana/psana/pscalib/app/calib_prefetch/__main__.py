import argparse
import logging
import signal
import sys

from psana.pscalib.app.calib_prefetch.calib_utils import CalibSource
from psana.utils import Logger


def main():
    """
    Entry point for the calibration prefetcher CLI tool.

    Parses command-line arguments, sets up logging and signal handlers,
    and initializes the CalibSource to fetch calibration constants.
    """
    parser = argparse.ArgumentParser(description="Calibration Prefetcher")
    parser.add_argument('-e', '--expcode', default=None, help='Experiment code')
    parser.add_argument('--xtc-dir', default=None, help='Optional path to XTC directory')
    parser.add_argument('--output-dir', default='/dev/shm', help='Output directory')
    parser.add_argument('--log-level', default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument('--timestamp', action='store_true', help='Include timestamp in log messages')
    parser.add_argument('--shmem', default=None, help='Shmem ID to run in shared memory mode')
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=[],
        help="List of detector names to prefetch calibration constants for."
    )
    parser.add_argument('--check-before-update', action='store_true', help='Only update if detector info has changed')
    parser.add_argument('--log-file', default=None, help='Optional log file path')
    args = parser.parse_args()

    log = Logger(timestamp=args.timestamp)
    log.logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    if args.log_file:
        handler = logging.FileHandler(args.log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        log.logger.addHandler(handler)

    def signal_handler(sig, frame):
        log.info("Shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    calib_source = CalibSource(
        expcode=args.expcode,
        xtc_dir=args.xtc_dir,
        output_dir=args.output_dir,
        shmem=args.shmem,
        detectors=args.detectors,
        check_before_update=args.check_before_update,
        log=log,
    )
    calib_source.run_loop()

if __name__ == '__main__':
    main()
