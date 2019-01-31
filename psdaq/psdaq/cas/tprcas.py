import sys

from pcaspy import SimpleServer, Driver
import time
from datetime import datetime
#import thread
import subprocess
import argparse
#import socket
#import json
import pdb

class myDriver(Driver):
    def __init__(self):
        super(myDriver, self).__init__()


def printDb(prefix):
    global pvdb

    print('=========== Serving %d PVs ==============' % len(pvdb))
    for key in sorted(pvdb):
        print(prefix+key)
    print('=========================================')
    return

if __name__ == '__main__':
    global pvdb
    pvdb = {}     # start with empty dictionary
    global prefix
    prefix = ''

    parser = argparse.ArgumentParser(prog=sys.argv[0], description='host PVs for TPR')

    parser.add_argument('-P', required=True, help='e.g. SXR or CXI:0 or CXI:1', metavar='PARTITION')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')

    args = parser.parse_args()
    myDriver.verbose = args.verbose

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

    server = SimpleServer()

    server.createPV(args.P, pvdb)
        
    driver = myDriver()

    try:
        # process CA transactions
        while True:
            server.process(0.1)
    except KeyboardInterrupt:
        print('\nInterrupted')
