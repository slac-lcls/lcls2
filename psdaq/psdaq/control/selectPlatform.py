#!/usr/bin/env python
"""
selectPlatform command
"""
from psdaq.control.DaqControl import DaqControl
from psdaq.control.control import detector_name
import pprint
import argparse

def common_match(alias, arglist):
    return arglist is not None and alias in arglist

def drp_match(alias, arglist):
    if arglist is not None:
        for arg in arglist:
            if (arg == alias) or (arg == detector_name(alias)):
                return True
    return False

def main():

    # Process arguments
    parser = argparse.ArgumentParser(epilog='For multisegment detector, specify drp alias without _N suffix.')
    parser.add_argument('-p', metavar='PLATFORM', type=int, choices=range(0, 8), default=0, help='platform (default 0)')
    parser.add_argument('-C', metavar='COLLECT_HOST', default='localhost', help='collection host (default localhost)')
    parser.add_argument('-t', type=int, metavar='TIMEOUT', default=2000,
                        help='timeout msec (default 2000)')
    parser.add_argument('-R', metavar='READOUT_GROUP', type=int, choices=range(0, 8), help='readout group (0-7, default platform)')
    parser.add_argument('-s', metavar='SELECT', action='append', help='select one alias (may be repeated)')
    parser.add_argument('--select-all', action='store_true', help='select all', dest='select_all')
    parser.add_argument('-u', metavar='UNSELECT', action='append', help='unselect one alias (may be repeated)')
    args = parser.parse_args()

    platform = args.p
    if args.R is None:
        readout = platform
    else:
        readout = args.R

    # instantiate DaqControl object
    control = DaqControl(host=args.C, platform=args.p, timeout=args.t)

    try:
        body = control.getPlatform()
    except Exception as ex:
        print('getPlatform(): %s' % ex)
    except KeyboardInterrupt:
        pass
    else:
        changed = False
        try:
            for level in body:
                for k, v in body[level].items():
                    alias = v['proc_info']['alias']

                    # select ...
                    if level == 'drp':
                        if args.select_all or drp_match(alias, args.s):
                            if v['active'] != 1 or v['det_info']['readout'] != readout:
                                changed = True
                                v['active'] = 1
                                v['det_info']['readout'] = readout
                    else:
                        if args.select_all or common_match(alias, args.s):
                            if v['active'] != 1:
                                changed = True
                                v['active'] = 1

                    # unselect ...
                    match = False
                    if level == 'drp':
                        if drp_match(alias, args.u):
                            match = True
                    else:
                        if common_match(alias, args.u):
                            match = True
                    if match:
                        if v['active'] != 0:
                            changed = True
                            v['active'] = 0
        except Exception:
            pprint.pprint(body)
            raise
        else:
            if changed:
                try:
                    retval = control.selectPlatform(body)
                except Exception as ex:
                    print('selectPlatform(): %s' % ex)
                except KeyboardInterrupt:
                    pass
                else:
                    if 'err_info' in retval:
                        print('Error: %s' % retval['err_info'])

if __name__ == '__main__':
    main()
