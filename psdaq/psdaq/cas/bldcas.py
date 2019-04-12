import sys
import logging

from psdaq.epicstools.PVAServer import PVAServer
import argparse

def printDb():
    global pvdb
    global prefix

    print('=========== Serving %d PVs ==============' % len(pvdb))
    for key in sorted(pvdb):
        print(prefix+key)
    print('=========================================')
    return

def main():
    global pvdb
    pvdb = {}     # start with empty dictionary
    global prefix
    prefix = ''

    parser = argparse.ArgumentParser(prog=sys.argv[0], description='host PVs for HPS diagnostic bus')

    parser.add_argument('-P', required=True, help='DAQ:SIM', metavar='PREFIX')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    stationstr = ''
    prefix = args.P+':'

    # Base PVs
    #   Each BLD service has
    #     a name
    #     a multicast address for data delivery
    #     a structure description of the payload
    #       field names, types, sizes
    pvdb['HPSEXAMPLE:ADDR'   ] = {'type'  : 'int', 'value' : 0xefff0100}
    pvdb['HPSEXAMPLE:PORT'   ] = {'type'  : 'int', 'value' : 0x2100}
    pvdb['HPSEXAMPLE:PAYLOAD'] = {'type'  : [('value', 'I'),],
                                  'value' : {'value':42,}}
    pvdb['HPSEXAMPLE:STRUCTID'] = {'type'  : 'string', 'value' : ''}

    # printDb(pvdb, prefix)
    printDb()

    server = PVAServer(__name__)
    server.createPV(prefix, pvdb)

    try:
        # process PVA transactions
        server.forever()
    except KeyboardInterrupt:
        print('\nInterrupted')



if __name__ == '__main__':
    main()
