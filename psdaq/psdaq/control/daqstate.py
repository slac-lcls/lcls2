#!/usr/bin/env python
"""
daqstate command
"""
from psdaq.control.DaqControl import DaqControl
from psdaq.control.ControlDef import ControlDef
import argparse
import json

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
    parser.add_argument('--phase1', metavar='JSON', default=None,
                        help='phase1Info (only use with --state)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--state', choices=ControlDef.states)
    group.add_argument('--transition', choices=ControlDef.transitions)
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
    try:
        instr = control.getInstrument()
    except KeyboardInterrupt:
        instr = None

    if instr is None:
        exit('Error: failed to read instrument name')
    elif instr != args.P:
        exit('Error: instrument name \'%s\' does not match \'%s\'' %
              (args.P, instr))

    if args.state:
        # change the state
        if args.phase1 is None:
            rv = control.setState(args.state)
        else:
            try:
                phase1 = json.loads(args.phase1)
            except Exception as ex:
                exit('Error: failed to parse JSON: %s' % ex)
            rv = control.setState(args.state, phase1)
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
            part1, part2, part3, part4, part5, part6, part7, part8 = control.monitorStatus()
            if part1 is None:
                break
            elif part1 == 'error':
                print('error: %s' % part2)
            elif part1 == 'warning':
                print('warning: %s' % part2)
            elif part1 == 'fileReport':
                print('data file: %s' % part2)
            elif part1 == 'step':
                print('step_done: %d' % part2)
            elif part1 == 'progress':
                print('progress: %s (%d/%d)' % (part2, part3, part4))
            elif part1 in ControlDef.transitions:
                print('transition: %-11s  state: %-11s  config: %s  recording: %s  bypass_activedet: %s  experiment_name: %s  run_number: %d  last_run_number: %d' %\
                      (part1, part2, part3, part4, part5, part6, part7, part8))
            else:
                print('unknown status: %s' % part1)

    else:
        # print current state
        transition, state, config_alias, recording, platform, bypass_activedet, \
            experiment_name, run_number, last_run_number = control.getStatus()
        print('last transition: %s  state: %s  configuration alias: %s  recording: %s  bypass_activedet: %s  experiment_name: %s  run_number: %d  last_run_number: %d' %\
              (transition, state, config_alias, recording, bypass_activedet, experiment_name, run_number, last_run_number))

if __name__ == '__main__':
    main()
