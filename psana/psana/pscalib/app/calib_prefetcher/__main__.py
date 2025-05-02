# __main__.py

import argparse
import logging  # needed to resolve string levels to constants
import signal
import sys
import time

from psana.utils import Logger

from .prefetcher import fetch_and_store


def main():
    parser = argparse.ArgumentParser(description="Calibration Prefetcher")
    parser.add_argument('-e', '--expcode', required=True, help='Experiment code')
    parser.add_argument('-r', '--runnum', required=True, type=int, help='Run number')
    parser.add_argument('--det-name', required=True, help='Detector name (e.g., jungfrau)')
    parser.add_argument('--dbsuffix', default='', help='Optional db suffix')
    parser.add_argument('--interval', type=int, default=5, help='Interval in minutes')
    parser.add_argument('--output-dir', default='/dev/shm', help='Output directory')
    parser.add_argument('--log-level', default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument('--rank', type=int, default=None, help='MPI rank (optional)')
    parser.add_argument('--timestamp', action='store_true', help='Include timestamp in log messages')
    parser.add_argument('--compress', action='store_true', help='Enable zstd compression')
    args = parser.parse_args()

    # Create logger with specified log level and options
    log = Logger(myrank=args.rank, timestamp=args.timestamp)
    log.logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    def signal_handler(sig, frame):
        log.info("Shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while True:
        try:
            fetch_and_store(
                expcode=args.expcode,
                runnum=args.runnum,
                det_name=args.det_name,
                dbsuffix=args.dbsuffix,
                output_dir=args.output_dir,
                log=log,
                compress=args.compress
            )
        except Exception as e:
            log.error(f"Error during fetch: {e}")
        time.sleep(args.interval * 60)

if __name__ == '__main__':
    main()

