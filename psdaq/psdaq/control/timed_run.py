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

    parser.add_argument('-p', type=int, choices=range(0, 8), default=1,
                        help='platform (default 1)')
    parser.add_argument('-C', metavar='COLLECT_HOST', default='daq-tst-dev03',
                        help='collection host (default daq-tst-dev03)')
    parser.add_argument('-t', type=int, metavar='TIMEOUT', default=10000,
                        help='timeout msec (default 10000)')
    parser.add_argument('-v', action='store_true', help='be verbose')
    parser.add_argument('--duration', type=int, default=10,
                        help='run duration seconds (default 10)')
    args = parser.parse_args()

    if args.rix:
      print('RIX ...')  # FIXME
    elif args.tmo:
      print('TMO ...')  # FIXME

    # instantiate DaqControl object
    control = DaqControl(host=args.C, platform=args.p, timeout=args.t)

    # configure logging handlers
    if args.v:
        level=logging.DEBUG
    else:
        level=logging.WARNING
    logging.basicConfig(level=level)
    logging.info('logging initialized')

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
