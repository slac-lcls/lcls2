#!/usr/bin/env python
"""
selectPlatform command
"""
from psdaq.control.collection import DaqControl
import pprint
import argparse

def main():

    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, choices=range(0, 8), default=0, help='platform (default 0)')
    parser.add_argument('-C', metavar='COLLECT_HOST', default='localhost', help='collection host')
    parser.add_argument('-t', type=int, metavar='TIMEOUT', default=2000,
                        help='timeout msec (default 2000)')
    parser.add_argument('-s', metavar='LEVEL/PID/HOST', action='append', help='select (may be repeated)')
    parser.add_argument('-u', metavar='LEVEL/PID/HOST', action='append', help='unselect (may be repeated)')
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
                    host = v['proc_info']['host']
                    pid = v['proc_info']['pid']
                    match = "%s/%s/%s" % (level, pid, host)
                    if args.s is not None:
                        # select
                        if match in args.s and v['active'] == 0:
                            changed = True
                            v['active'] = 1
                    if args.u is not None:
                        # unselect
                        if match in args.u and v['active'] == 1:
                            changed = True
                            v['active'] = 0
        except KeyError as ex:
            print('Error: failed to parse reply: %s' % ex)
            pprint.pprint(body)
        else:
            if changed:
                try:
                    control.selectPlatform(body)
                except Exception as ex:
                    print('selectPlatform(): %s' % ex)
                except KeyboardInterrupt:
                    pass

if __name__ == '__main__':
    main()
