import sys
import logging

from psdaq.epicstools.PVAServer import PVAServer
import time
from datetime import datetime
#import thread
import subprocess
import argparse
#import socket
#import json
import pdb

def printDb(prefix):
    global pvdb

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

    parser = argparse.ArgumentParser(prog=sys.argv[0], description='host PVs for TPR')

    parser.add_argument('-P', required=True, help='e.g. SXR or CXI:0 or CXI:1', metavar='PARTITION')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    #
    # Parse the PARTITION argument for the instrument name and station #.
    # If the partition name includes a colon, PV names will include station # even if 0.
    # If no colon is present, station # defaults to 0 and is not included in PV names.
    # Partition names 'AMO' and 'AMO:0' thus lead to different PV names.
    #

    # PVs
    pvdb[prefix+':ACCSEL' ] = {'type' : 'int', 'value': 0}
    pvdb[prefix+':LINKSTATE' ] = {'type' : 'int', 'value': 0}
    pvdb[prefix+':LINKLATCH' ] = {'type' : 'int', 'value': 0}
    pvdb[prefix+':RXERRS' ] = {'type' : 'int', 'value': 0}
    pvdb[prefix+':RXPOL'  ] = {'type' : 'int', 'value': 0}
    pvdb[prefix+':FRAMERATE' ] = {'type' : 'float', 'value': 0}
    pvdb[prefix+':RXCLKRATE' ] = {'type' : 'float', 'value': 0}

    pvdb[prefix+':IRQENA' ] = {'type' : 'int', 'value': 0}
    pvdb[prefix+':EVTCNT' ] = {'type' : 'int', 'value': 0}

    for i in range(12):
        prefix = ':CH%u'%i
        pvdb[prefix+':MODE'  ] = {'type' : 'int', 'value': 0}
        pvdb[prefix+':DELAY' ] = {'type' : 'float', 'value': 0}
        pvdb[prefix+':WIDTH' ] = {'type' : 'float', 'value': 5.4e-9}
        pvdb[prefix+':POL'   ] = {'type' : 'int', 'value': 0}
        pvdb[prefix+':DSTSEL'] = {'type' : 'int', 'value': 0}
        pvdb[prefix+':DESTNS'] = {'type' : 'int', 'value': 0}
        pvdb[prefix+':RSEL'  ] = {'type' : 'int', 'value': 0}
        pvdb[prefix+':FRATE' ] = {'type' : 'int', 'value': 0}
        pvdb[prefix+':ARATE' ] = {'type' : 'int', 'value': 0}
        pvdb[prefix+':ATS'   ] = {'type' : 'int', 'value': 1}
        pvdb[prefix+':SEQIDX'] = {'type' : 'int', 'value': 0}
        pvdb[prefix+':SEQBIT'] = {'type' : 'int', 'value': 0}
        pvdb[prefix+':XPART' ] = {'type' : 'int', 'value': 0}
        pvdb[prefix+':BSTART'] = {'type' : 'int', 'value': 0}
        pvdb[prefix+':BWIDTH'] = {'type' : 'int', 'value': 1}
        pvdb[prefix+':RATE'  ] = {'type' : 'float', 'value': 0}

    for i in range(12,14):
        prefix = ':CH%u'%i
        pvdb[prefix+':MODE'  ] = {'type' : 'int', 'value': 0}
#        pvdb[prefix+':DELAY' ] = {'type' : 'float', 'value': 0}
#        pvdb[prefix+':WIDTH' ] = {'type' : 'float', 'value': 5.4e-9}
#        pvdb[prefix+':POL'   ] = {'type' : 'int', 'value': 0}
        pvdb[prefix+':DSTSEL'] = {'type' : 'int', 'value': 0}
        pvdb[prefix+':DESTNS'] = {'type' : 'int', 'value': 0}
        pvdb[prefix+':RSEL'  ] = {'type' : 'int', 'value': 0}
        pvdb[prefix+':FRATE' ] = {'type' : 'int', 'value': 0}
        pvdb[prefix+':ARATE' ] = {'type' : 'int', 'value': 0}
        pvdb[prefix+':ATS'   ] = {'type' : 'int', 'value': 1}
        pvdb[prefix+':SEQIDX'] = {'type' : 'int', 'value': 0}
        pvdb[prefix+':SEQBIT'] = {'type' : 'int', 'value': 0}
        pvdb[prefix+':XPART' ] = {'type' : 'int', 'value': 0}
        pvdb[prefix+':BSTART'] = {'type' : 'int', 'value': 0}
        pvdb[prefix+':BWIDTH'] = {'type' : 'int', 'value': 1}
        pvdb[prefix+':RATE'  ] = {'type' : 'float', 'value': 0}

    printDb(args.P)

    server = PVAServer(__name__)
    server.createPV(args.P, pvdb)
    server.forever()

if __name__ == '__main__':
    main()
