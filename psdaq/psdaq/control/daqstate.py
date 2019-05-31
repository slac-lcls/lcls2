#!/usr/bin/env python
"""
daqstate command
"""
from psdaq.control.collection import DaqControl
import argparse

def main():

    # process arguments
    parser = argparse.ArgumentParser(
        epilog='--config or -B required with --state or --transition')
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
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('--config', metavar='ALIAS', help='configuration alias')
    group2.add_argument('-B', action="store_true",
                        help='shortcut for --config BEAM')
    args = parser.parse_args()

    config = None
    if args.B:
        config = 'BEAM'
    elif args.config:
        config = args.config

    if (args.state or args.transition) and config is None:
        parser.error('--config or -B required with --state or --transition')

    # instantiate DaqControl object
    control = DaqControl(host=args.C, platform=args.p, timeout=args.t)

    if args.state:
        # change the state
        rv = control.setState(args.state, config_alias=config)
        if rv is not None:
            print('Error: %s' % rv)

    elif args.transition:
        # transition request
        rv = control.setTransition(args.transition, config_alias=config)
        if rv is not None:
            print('Error: %s' % rv)

    elif args.monitor:
        # monitor the status
        while True:
            part1, part2 = control.monitorStatus()
            if part1 is None:
                break
            elif part1 == 'error':
                print('error: %s' % part2)
            else:
                print('transition: %-10s  state: %s' % (part1, part2))

    else:
        # print current state
        transition, state = control.getStatus()
        print('last transition: %s  state: %s' % (transition, state))

if __name__ == '__main__':
    main()
