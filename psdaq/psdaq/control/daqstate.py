#!/usr/bin/env python
"""
daqstate command
"""
from psdaq.control.control import DaqControl
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
    group.add_argument('--config', metavar='ALIAS', help='configuration alias')
    group.add_argument('-B', action="store_true", help='shortcut for --config BEAM')
    args = parser.parse_args()

    config = None
    if args.config:
        config = args.config
    elif args.B:
        config = "BEAM"

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

    elif config:
        # config alias request
        rv = control.setConfig(config)
        if rv is not None:
            print('Error: %s' % rv)

    elif args.monitor:
        # monitor the status
        while True:
            part1, part2, part3 = control.monitorStatus()
            if part1 is None:
                break
            elif part1 == 'error':
                print('error: %s' % part2)
            else:
                print('transition: %-11s  state: %-11s  config: %s' % (part1, part2, part3))

    else:
        # print current state
        transition, state, config_alias = control.getStatus()
        print('last transition: %s  state: %s  configuration alias: %s' % (transition, state, config_alias))

if __name__ == '__main__':
    main()
