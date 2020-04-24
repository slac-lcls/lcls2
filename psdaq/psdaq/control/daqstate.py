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
    parser.add_argument('-P', metavar='INSTRUMENT', required=True,
                        help='instrument name (required)')
    parser.add_argument('-C', metavar='COLLECT_HOST', default='localhost',
                        help='collection host (default localhost)')
    parser.add_argument('-t', type=int, metavar='TIMEOUT', default=10000,
                        help='timeout msec (default 10000)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--state', choices=DaqControl.states)
    group.add_argument('--transition', choices=DaqControl.transitions)
    group.add_argument('--monitor', action="store_true")
    group.add_argument('--config', metavar='ALIAS', help='configuration alias')
    group.add_argument('--record', type=int, choices=range(0, 2), help='recording flag')
    group.add_argument('--bypass', type=int, choices=range(0, 2), help='bypass active detectors file flag')
    group.add_argument('-B', action="store_true", help='shortcut for --config BEAM')
    args = parser.parse_args()

    config = None
    if args.config:
        config = args.config
    elif args.B:
        config = "BEAM"

    # instantiate DaqControl object
    control = DaqControl(host=args.C, platform=args.p, timeout=args.t)

    # verify instrument name match
    instr = control.getInstrument()
    if instr is None:
        exit('Error: failed to read instrument name')
    elif instr != args.P:
        exit('Error: instrument name \'%s\' does not match \'%s\'' %
              (args.P, instr))

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

    elif args.record is not None:
        # recording flag request
        if args.record == 0:
            rv = control.setRecord(False)
        else:
            rv = control.setRecord(True)
        if rv is not None:
            print('Error: %s' % rv)

    elif args.bypass is not None:
        # active detectors file bypass flag request
        if args.bypass == 0:
            rv = control.setBypass(False)
        else:
            rv = control.setBypass(True)
        if rv is not None:
            print('Error: %s' % rv)

    elif args.monitor:
        # monitor the status
        while True:
            part1, part2, part3, part4 = control.monitorStatus()
            if part1 is None:
                break
            elif part1 == 'error':
                print('error: %s' % part2)
            elif part1 == 'fileReport':
                print('data file: %s' % part2)
            elif part1 == 'progress':
                print('progress: %s (%d/%d)' % (part2, part3, part4))
            else:
                print('transition: %-11s  state: %-11s  config: %s  recording: %s' % (part1, part2, part3, part4))

    else:
        # print current state
        transition, state, config_alias, recording, platform = control.getStatus()
        print('last transition: %s  state: %s  configuration alias: %s  recording: %s' % (transition, state, config_alias, recording))

if __name__ == '__main__':
    main()
