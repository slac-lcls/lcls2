#!/usr/bin/env python
"""
showPlatform command
"""
from psdaq.control.DaqControl import DaqControl
import pprint
import argparse
from operator import itemgetter

def main():

    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, choices=range(0, 8), default=0, help='platform (default 0)')
    parser.add_argument('-C', metavar='COLLECT_HOST', default='localhost', help='collection host')
    parser.add_argument('-t', type=int, metavar='TIMEOUT', default=2000,
                        help='timeout msec (default 2000)')
    parser.add_argument('-v', action='store_true', help='be verbose')
    parser.add_argument('--json', action='store_true', help='JSON configuration')
    args = parser.parse_args()
    platform = args.p

    # instantiate DaqControl object
    control = DaqControl(host=args.C, platform=args.p, timeout=args.t)

    # get instrument/hutch name
    instrument = None
    try:
        instrument = control.getInstrument()
    except Exception as ex:
        print('getInstrument() Exception: %s' % ex)
    if instrument is None:
        return

    if args.json:
        try:
            body = control.getJsonConfig()
        except Exception as ex:
            print('getJsonConfig() Exception: %s' % ex)
        else:
            print(body)
    else:
        try:
            body = control.getPlatform()
        except Exception as ex:
            print('getPlatform() Exception: %s' % ex)
        else:
            displayList = []
            try:
                for level in body:
                    for k, v in body[level].items():
                        host = v['proc_info']['host']
                        if v['active'] == 1:
                            host = host + ' *'
                        alias = v['proc_info']['alias']
                        pid = v['proc_info']['pid']
                        if level == 'drp' and v['active'] == 1:
                            display_tuple = (level, alias,
                                             "%-16s %s/%s/%-16s\n%42s: %s" % \
                                             (alias, level, pid, host,       \
                                              'readout group', v['det_info']['readout']))
                        else:
                            display_tuple = (level, alias,
                                             "%-16s %s/%s/%-16s" % (alias, level, pid, host))
                        displayList.append(display_tuple)
            except Exception:
                print('----- body -----')
                pprint.pprint(body)
                raise
            else:
                if args.v:
                    print('getPlatform() reply:')
                    pprint.pprint(body)
                print("Partition|         Node")
                print("id/name  | alias            level/pid/host (* = active)")
                print("---------+-----------------------------------------------------")
                print("%s/%-8s " % (platform, instrument), end='')
                firstLine = True
                for nn in sorted(displayList, key=itemgetter(0,1)):
                    if firstLine:
                        print(nn[2])
                        firstLine = False
                    else:
                        print("          ", nn[2])
                if firstLine:
                    print()

if __name__ == '__main__':
    main()
