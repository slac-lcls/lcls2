import sys

from pcaspy import SimpleServer, Driver
import time
from datetime import datetime
import argparse
#import socket
#import json
import pdb

NLanes = 4
NApps = 4
WfLen = 1024

class myDriver(Driver):
    def __init__(self):
        super(myDriver, self).__init__()


def printDb():
    global pvdb
    global prefix

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

    parser = argparse.ArgumentParser(prog=sys.argv[0], description='host PVs for PGP')

    parser.add_argument('-P', required=True, help='DAQ:SIM', metavar='PREFIX')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')

    args = parser.parse_args()
    myDriver.verbose = args.verbose

    stationstr = ''
    prefix = args.P+':'

    # Base PVs
    #  Range of trigger delay setting capability
    pvdb[stationstr+'BASE:INTTRIGRANGE' ] = {'type' : 'int', 
                                             'count' : 2, 
                                             'value' : [0,0x0fffffff] }
    #  Clock Freq [MHz] of trigger delay setting
    pvdb[stationstr+'BASE:INTTRIGCLK'   ] = {'type' : 'float', 
                                             'value' : 156.25 }
    #  Trigger delay setting (determined by XPM)
    pvdb[stationstr+'BASE:INTTRIGVAL'   ] = {'type' : 'int', 
                                             'value' : 0 }
    #  Trigger delay target (tuned to sync with beam)
    pvdb[stationstr+'BASE:ABSTRIGTARGET'] = {'type' : 'float', 
                                             'value' : 95 }
    #  Internal event pipeline depth
    pvdb[stationstr+'BASE:INTPIPEDEPTH' ] = {'type' : 'int', 
                                             'value' : 0 }
    #  Internal event pipeline almost full value
    pvdb[stationstr+'BASE:INTAFULLVAL'  ] = {'type' : 'int', 
                                             'value' : 0 }
    #  Minimum L0 trigger spacing requirement
    pvdb[stationstr+'BASE:MINL0INTERVAL'] = {'type' : 'float', 
                                             'value' : 4.2 }
    #  Upstream round trip time (XPM generates L0 -> XPM receives almost full from device)
    pvdb[stationstr+'BASE:UPSTREAMRTT'  ] = {'type' : 'float', 
                                             'value' : 0.8 }
    #  Upstream round trip time (XPM generates L0 -> XPM receives almost full from DRP)
    pvdb[stationstr+'BASE:DNSTREAMRTT'  ] = {'type' : 'float', 
                                             'value' : 1.2 }
    #  Upstream round trip time (XPM generates L0 -> XPM receives almost full from DRP)
    pvdb[stationstr+'BASE:PARTITION'    ] = {'type' : 'int', 
                                             'value' : 0 }
    #  This PV triggers execution of all configuration parameters
    pvdb[stationstr+'BASE:APPLYCONFIG'  ] = {'type' : 'int', 
                                             'value' : 0 }
    pvdb[stationstr+'BASE:UNDOCONFIG'   ] = {'type' : 'int', 
                                             'value' : 0 }

    # Specific PVs
    pvdb[stationstr+'ENABLE'   ] = {'type' : 'int', 
                                    'count': 4,
                                    'value' : [0]*4 }
    pvdb[stationstr+'RAW_GATE' ] = {'type' : 'int', 
                                    'count': 4,
                                    'value' : [200]*4 }
    pvdb[stationstr+'RAW_PS'   ] = {'type' : 'int', 
                                    'count': 4,
                                    'value' : [1]*4 }
    pvdb[stationstr+'FEX_GATE' ] = {'type' : 'int', 
                                    'count': 4,
                                    'value' : [200]*4 }
    pvdb[stationstr+'FEX_PS'   ] = {'type' : 'int', 
                                    'count': 4,
                                    'value' : [1]*4 }
    pvdb[stationstr+'FEX_YMIN' ] = {'type' : 'int', 
                                    'count': 4,
                                    'value' : [508]*4 }
    pvdb[stationstr+'FEX_YMAX' ] = {'type' : 'int', 
                                    'count': 4,
                                    'value' : [516]*4 }
    pvdb[stationstr+'FEX_XPRE' ] = {'type' : 'int', 
                                    'count': 4,
                                    'value' : [2]*4 }
    pvdb[stationstr+'FEX_XPOST'] = {'type' : 'int', 
                                    'count': 4,
                                    'value' : [3]*4 }

    pvdb[stationstr+'RESET'  ] = {'type' : 'int', 
                                  'value' : 0 }
    pvdb[stationstr+'PGPLOOPBACK'  ] = {'type' : 'int', 
                                        'value' : 0 }
    pvdb[stationstr+'TESTPATTERN'  ] = {'type' : 'int', 
                                        'value' : -1 }
    pvdb[stationstr+'TESTPATTERR'  ] = {'type' : 'int', 
                                        'count' : 4,
                                        'value' : [0]*4 }
    pvdb[stationstr+'TESTPATTBIT'  ] = {'type' : 'int',
                                        'count' : 4,
                                        'value' : [0]*4 }
    pvdb[stationstr+'WRFIFOCNT'  ] = {'type' : 'int', 
                                      'count' : 4,
                                      'value' : [0]*4 }
    pvdb[stationstr+'RDFIFOCNT'  ] = {'type' : 'int',
                                      'count' : 4,
                                      'value' : [0]*4 }

    # Status Monitoring

    # Timing link frames
    pvdb[stationstr+'TIMFRAMECNT'  ] = {'type' : 'int', 
                                        'value' : 0 }
    # Timing link cycles paused (deadtime)
    pvdb[stationstr+'TIMPAUSECNT'  ] = {'type' : 'int', 
                                        'value' : 0 }
    # PGP link status
    pvdb[stationstr+'PGPLOCLINKRDY'] = {'type' : 'int',
                                        'count': 4,
                                        'value' : [0]*4 }
    pvdb[stationstr+'PGPREMLINKRDY'] = {'type' : 'int',
                                        'count': 4,
                                        'value' : [0]*4 }
    # PGP reference clocks
    pvdb[stationstr+'PGPTXCLKFREQ' ] = {'type' : 'int',
                                        'count': 4,
                                        'value' : [0]*4 }
    pvdb[stationstr+'PGPRXCLKFREQ' ] = {'type' : 'int',
                                        'count': 4,
                                        'value' : [0]*4 }
    # PGP frames transmitted
    pvdb[stationstr+'PGPTXCNT'     ] = {'type' : 'int',
                                        'count': 4,
                                        'value' : [0]*4 }
    # PGP frames error in transmission
    pvdb[stationstr+'PGPTXERRCNT'  ] = {'type' : 'int',
                                        'count': 4,
                                        'value' : [0]*4 }
    # PGP frames received (deadtime link)
    pvdb[stationstr+'PGPRXCNT'     ] = {'type' : 'int',
                                        'count': 4,
                                        'value' : [0]*4 }
    # Last PGP opcode received
    pvdb[stationstr+'PGPRXLAST'    ] = {'type' : 'int',
                                        'count': 4,
                                        'value' : [0]*4 }
    # Bytes(?) free in buffer pool
    pvdb[stationstr+'RAW_FREEBUFSZ'] = {'type' : 'int',
                                        'count': 4,
                                        'value' : [0]*4 }
    # Events free in buffer pool
    pvdb[stationstr+'RAW_FREEBUFEVT'] = {'type' : 'int',
                                        'count': 4,
                                        'value' : [0]*4 }
    # Acquisition state of buffers
    pvdb[stationstr+'RAW_BUFSTATE'     ] = {'type' : 'int',
                                        'count': 16,
                                        'value' : [0]*16 }
    # Trigger state of buffers
    pvdb[stationstr+'RAW_TRGSTATE'     ] = {'type' : 'int',
                                        'count': 16,
                                        'value' : [0]*16 }
    # Beginning address of buffers
    pvdb[stationstr+'RAW_BUFBEG'       ] = {'type' : 'int',
                                        'count': 16,
                                        'value' : [0]*16 }
    # Ending address of buffers
    pvdb[stationstr+'RAW_BUFEND'       ] = {'type' : 'int',
                                        'count': 16,
                                        'value' : [0]*16 }
    # Bytes(?) free in buffer pool
    pvdb[stationstr+'FEX_FREEBUFSZ'] = {'type' : 'int',
                                        'count': 4,
                                        'value' : [0]*4 }
    # Events free in buffer pool
    pvdb[stationstr+'FEX_FREEBUFEVT'] = {'type' : 'int',
                                        'count': 4,
                                        'value' : [0]*4 }

    # Data monitoring
    pvdb[stationstr+'RAWDATA'] = {'type' : 'int',
                                  'count': WfLen,
                                  'value' : [512]*WfLen }
    pvdb[stationstr+'FEXDATA'] = {'type' : 'int',
                                  'count': WfLen,
                                  'value' : [512]*WfLen }

    # printDb(pvdb, prefix)
    printDb()

    server = SimpleServer()

    server.createPV(prefix, pvdb)
    driver = myDriver()

    try:
        # process CA transactions
        while True:
            server.process(0.1)
    except KeyboardInterrupt:
        print('\nInterrupted')
