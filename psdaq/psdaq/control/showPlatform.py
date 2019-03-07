#!/usr/bin/env python
"""
showPlatform command
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
    parser.add_argument('-v', action='store_true', help='be verbose')
    args = parser.parse_args()
    platform = args.p

    # instantiate DaqControl object
    control = DaqControl(host=args.C, platform=args.p, timeout=args.t)

    partName = '(None)'
    try:
        body = control.getPlatform()
    except Exception as ex:
        print('Exception: %s' % ex)
    except KeyboardInterrupt:
        pass
    else:
        displayList = []
        try:
            for level in body:
                for k, v in body[level].items():
                    host = v['proc_info']['host']
                    if v['active'] == 1:
                        host = host + ' *'
                    pid = v['proc_info']['pid']
                    display = "%s/%s/%-16s" % (level, pid, host)
                    if level == 'control':
                        # show control level first
                        displayList.insert(0, display)
                    else:
                        displayList.append(display)
        except KeyError as ex:
            print('Error: failed to parse reply: %s' % ex)
            pprint.pprint(body)
        else:
            if args.v:
                print('getPlatform() reply:')
                pprint.pprint(body)
            print("Platform | Partition      |    Node")
            print("         | id/name        | level/pid/host (* = active)")
            print("---------+----------------+------------------------------------")
            print("  %3s     %2s/%-12s" % (platform, platform, partName), end='')
            firstLine = True
            for nn in displayList:
                if firstLine:
                    print("  ", nn)
                    firstLine = False
                else:
                    print("                           ", nn)
            if firstLine:
                print()

if __name__ == '__main__':
    main()
