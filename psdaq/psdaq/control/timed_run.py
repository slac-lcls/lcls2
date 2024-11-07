#!/usr/bin/env python

import sys
import os
import logging
import threading
from psdaq.control.DaqControl import DaqControl
from psdaq.control.TimedRun import TimedRun
import argparse

#
# deduce_platform3 - deduce platform and collection host
#
# RETURNS: Two values: platform number (or -1 on error) and collection host name
#
def deduce_platform3(configfilename, platform=None):
    platform_rv = -1  # return -1 on error
    cc = {'platform': platform, 'procmgr_config': None, 'TESTRELDIR': '', 'CONDA_PREFIX': os.environ['CONDA_PREFIX'],
          'CONFIGDIR': '', 'collect_host': '',
          'id': 'id', 'cmd': 'cmd', 'flags': 'flags', 'port': 'port', 'host': 'host', '__file__': configfilename,
          'rtprio': 'rtprio', 'env': 'env', 'evr': 'evr', 'conda': 'conda', 'procmgr_macro': {}}
    try:
        exec(compile(open(configfilename).read(), configfilename, 'exec'), {}, cc)
        collect_host_rv = cc['collect_host']
        if type(cc['platform']) == type('') and cc['platform'].isdigit():
            platform_rv = int(cc['platform'])
    except:
        print('deduce_platform3 Error:', sys.exc_info()[1])

    return platform_rv, collect_host_rv

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', help='configuration file name (required)', required=True)

    parser.add_argument('--duration', type=int, metavar='DURATION', required=True,
                        help='duration seconds (required)')

    parser.add_argument('-t', type=int, metavar='TIMEOUT', default=10000,
                        help='timeout msec (default 10000)')

    parser.add_argument('-v', action='store_true', help='be verbose')

    args = parser.parse_args()

    # configure logging handlers
    if args.v:
        level=logging.DEBUG
    else:
        level=logging.WARNING
    logging.basicConfig(level=level)
    logging.info('logging initialized')

    # instantiate DaqControl object
    platform, collect_host = deduce_platform3(args.config)
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
