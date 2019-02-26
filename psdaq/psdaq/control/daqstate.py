#!/usr/bin/env python
"""
daqstate command
"""
from psdaq.control.collection import DaqControl, MonitorThread, SignalHandler
import argparse
import logging
import threading
import signal
import time

def monitor_callback(msg):
    logging.debug("entered monitor_callback()")

    try:
        if msg['header']['key'] == 'status':
            print('transition: %-10s  state: %s' %
                (msg['body']['transition'], msg['body']['state']))

        elif msg['header']['key'] == 'error':
            print('error: %s' % msg['body']['error'])

    except KeyError as ex:
        logging.error('monitor_callback() KeyError: %s' % ex)

def main():
    global monitor_callback

    # process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, choices=range(0, 8), default=0,
                        help='platform (default 0)')
    parser.add_argument('-C', metavar='COLLECT_HOST', default='localhost',
                        help='collection host (default localhost)')
    parser.add_argument('-t', type=int, metavar='TIMEOUT', default=10000,
                        help='timeout msec (default 10000)')
    parser.add_argument('-v', action='store_true', help='be verbose')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--state', choices=DaqControl.states)
    group.add_argument('--transition', choices=DaqControl.transitions)
    group.add_argument('--monitor', action="store_true")
    args = parser.parse_args()

    # configure logging
    if args.v:
        logging.basicConfig(level=logging.DEBUG, format='%(threadName)s: %(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(threadName)s: %(asctime)s - %(levelname)s - %(message)s')

    # instantiate DaqControl object
    control = DaqControl(host=args.C, platform=args.p, timeout=args.t)

    if args.state:
        # change the state
        rv = control.setState(args.state)
        if rv is not None:
            print('Error: %s' % rv)

    elif args.transition:
        # transition request
        rv = control.setTransition(args.transition)
        if rv is not None:
            print('Error: %s' % rv)

    elif args.monitor:
        # create event for stopping threads
        stopper = threading.Event()

        # create worker thread for monitoring zmq socket
        worker = MonitorThread(monitor_callback, stopper, args.C, args.p)

        # create our signal handler and connect it
        handler = SignalHandler(stopper)
        signal.signal(signal.SIGINT, handler)

        # start worker thread
        worker.start()

        # in main thread, just wait for stopper event
        stopper.wait()

        # shutting down: join worker thread
        worker.join()

    else:
        # print current state
        print(control.getState())

if __name__ == '__main__':
    main()
