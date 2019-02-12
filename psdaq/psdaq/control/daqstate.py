#!/usr/bin/env python
"""
daqstate command
"""
from psdaq.control.collection import DaqControl
import argparse

def main():

    # process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, choices=range(0, 8), default=0,
                        help='platform (default 0)')
    parser.add_argument('-C', metavar='COLLECT_HOST', default='localhost',
                        help='collection host (default localhost)')
    parser.add_argument('-t', type=int, metavar='TIMEOUT', default=10000,
                        help='timeout msec (default 10000)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--state', choices=DaqControl.states)
    group.add_argument('--transition', choices=DaqControl.transitions)
    group.add_argument('--monitor', action="store_true")
    args = parser.parse_args()

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
        # monitor the status
        while True:
            transition, state = control.monitorStatus()
            if transition is None:
                break
            print('transition: %-10s  state: %s' % (transition, state))

    else:
        # print current state
        print(control.getState())

if __name__ == '__main__':
    main()
