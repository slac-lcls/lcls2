import argparse
import logging
import signal
import sys
import time

from psana.utils import Logger

from .calib_utils import update_calib


def main():
    """
    Entry point for the calibration prefetcher tool.

    Parses command-line arguments, sets up logging and signal handling,
    and runs a loop to periodically update calibration constants using
    `update_calib()` based on the experiment's newest run.

    Args:
        None

    Returns:
        None
    """

    parser = argparse.ArgumentParser(description="Calibration Prefetcher")
    parser.add_argument('--xtc-dir', default=None, help='Optional path to XTC directory')
    parser.add_argument('-e', '--expcode', required=True, help='Experiment code')
    parser.add_argument('--interval', type=int, default=5, help='Interval in minutes')
    parser.add_argument('--output-dir', default='/dev/shm', help='Output directory')
    parser.add_argument('--log-level', default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument('--timestamp', action='store_true', help='Include timestamp in log messages')
    args = parser.parse_args()

    log = Logger(timestamp=args.timestamp)
    log.logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    def signal_handler(sig, frame):
        log.info("Shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while True:
        try:
            update_calib(args.expcode, output_dir=args.output_dir, log=log, xtc_dir=args.xtc_dir)
        except Exception as e:
            log.error(f"Error during update_calib: {e}")
        time.sleep(args.interval * 60)

if __name__ == '__main__':
    main()
