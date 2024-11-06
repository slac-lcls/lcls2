#!/usr/bin/env python

import sys
import logging
import threading
from psdaq.control.DaqControl import DaqControl
from psdaq.control.TimedRun import TimedRun
import argparse

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('--rix', action='store_true', help='RIX')
    group.add_argument('--tmo', action='store_true', help='TMO')

    parser.add_argument('-t', type=int, metavar='TIMEOUT', default=10000,
                        help='timeout msec (default 10000)')
    parser.add_argument('-v', action='store_true', help='be verbose')
    parser.add_argument('--duration', type=int, default=10,
                        help='run duration seconds (default 10)')
    args = parser.parse_args()

    # configure logging handlers
    if args.v:
        level=logging.DEBUG
    else:
        level=logging.WARNING
    logging.basicConfig(level=level)
    logging.info('logging initialized')

    # fill in collection host and platform number for the chosen hutch
    if args.rix:
        collect_host = 'drp-srcf-cmp004'
        platform = 0
        logging.debug(f'RIX: collect_host = {collect_host}  platform = {platform}')
    elif args.tmo:
        collect_host = 'drp-srcf-mon001'
        platform = 0
        logging.debug(f'TMO: collect_host = {collect_host}  platform = {platform}')

    # instantiate DaqControl object
    control = DaqControl(host=collect_host, platform=platform, timeout=args.t)

    # get initial DAQ state
    daqState = control.getState()
    logging.info('initial state: %s' % daqState)
    if daqState == 'error':
        sys.exit('failed to get initial DAQ state')

    # instantiate TimedRun
    run = TimedRun(control, daqState=daqState, args=args)

    run.stage()

    try:

        # -- begin script --------------------------------------------------------

        # run daq for the specified time duration
        run.set_running_state()
        run.sleep(args.duration)

        # -- end script ----------------------------------------------------------

    except KeyboardInterrupt:
        run.push_socket.send_string('shutdown') #shutdown the daq communicator thread
        sys.exit('interrupted')

    run.unstage()

    run.push_socket.send_string('shutdown') #shutdown the daq communicator thread

if __name__ == '__main__':
    main()
