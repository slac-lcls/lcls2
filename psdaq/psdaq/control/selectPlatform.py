#!/usr/bin/env python
"""
selectPlatform command
"""
from psdaq.control.collection import DaqControl
import pprint
import argparse

def main():

    # Process arguments
    parser = argparse.ArgumentParser(epilog='The -R argument is required when selecting drp.')
    parser.add_argument('-p', metavar='PLATFORM', type=int, choices=range(0, 8), default=0, help='platform (default 0)')
    parser.add_argument('-C', metavar='COLLECT_HOST', default='localhost', help='collection host (default localhost)')
    parser.add_argument('-t', type=int, metavar='TIMEOUT', default=2000,
                        help='timeout msec (default 2000)')
    parser.add_argument('-R', metavar='READOUT_GROUP', type=int, choices=range(0, 8), help='readout group (0-7) (drp only)')
    parser.add_argument('-s', metavar='SELECT', action='append', help='select one alias (may be repeated)')
    parser.add_argument('--select-all', action='store_true', help='select all', dest='select_all')
    parser.add_argument('-u', metavar='UNSELECT', action='append', help='unselect one alias (may be repeated)')
    args = parser.parse_args()

    platform = args.p

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

                    if args.select_all or (args.s is not None and alias in args.s):
                        if level == 'drp':
                            # select drp
                            if args.R is None:
                                # drp requires readout group
                                parser.error('select drp: requires -R')
                            if v['active'] != 1 or v['readout'] != args.R:
                                changed = True
                                v['active'] = 1
                                v['readout'] = args.R
                        else:
                            # select teb or meb
                            if v['active'] != 1:
                                changed = True
                                v['active'] = 1

                    if args.u is not None and alias in args.u:
                        # unselect drp or teb or meb
                        if v['active'] != 0:
                            changed = True
                            v['active'] = 0
                            if level == 'drp':
                                v['readout'] = platform
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
                    if 'error' in retval:
                        print('Error: %s' % retval['error'])

if __name__ == '__main__':
    main()
