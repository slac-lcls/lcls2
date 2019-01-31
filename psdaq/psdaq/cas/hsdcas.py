import sys
import logging

from psdaq.epicstools.PVAServer import PVAServer
import time
from datetime import datetime
import argparse
#import socket
#import json
import pdb

NLanes = 4
NApps = 4
WfLen = 1024
NChans = 4

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

    parser = argparse.ArgumentParser(prog=sys.argv[0], description='host PVs for PGP')

    parser.add_argument('-P', required=True, help='DAQ:SIM', metavar='PREFIX')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

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
    pvdb[stationstr+'BASE:ENABLETR'     ] = {'type' : 'int',
                                             'value' : 0 }
    pvdb[stationstr+'BASE:DISABLETR'    ] = {'type' : 'int',
                                             'value' : 0 }

    # Specific PVs
    pvdb[stationstr+'ENABLE'   ] = {'type' : 'int',
                                    'count': NChans,
                                    'value' : [1]*NChans }
    pvdb[stationstr+'RAW_START' ] = {'type' : 'int',
                                     'count': NChans,
                                     'value' : [NChans]*NChans }
    pvdb[stationstr+'RAW_GATE' ] = {'type' : 'int',
                                    'count': NChans,
                                    'value' : [200]*NChans }
    pvdb[stationstr+'RAW_PS'   ] = {'type' : 'int',
                                    'count': NChans,
                                    'value' : [1]*NChans }
    pvdb[stationstr+'FEX_START' ] = {'type' : 'int',
                                     'count': NChans,
                                     'value' : [NChans]*NChans }
    pvdb[stationstr+'FEX_GATE' ] = {'type' : 'int',
                                    'count': NChans,
                                    'value' : [200]*NChans }
    pvdb[stationstr+'FEX_PS'   ] = {'type' : 'int',
                                    'count': NChans,
                                    'value' : [1]*NChans }
    pvdb[stationstr+'FEX_YMIN' ] = {'type' : 'int',
                                    'count': NChans,
                                    'value' : [508]*NChans }
    pvdb[stationstr+'FEX_YMAX' ] = {'type' : 'int',
                                    'count': NChans,
                                    'value' : [516]*NChans }
    pvdb[stationstr+'FEX_XPRE' ] = {'type' : 'int',
                                    'count': NChans,
                                    'value' : [2]*NChans }
    pvdb[stationstr+'FEX_XPOST'] = {'type' : 'int',
                                    'count': NChans,
                                    'value' : [3]*NChans }
    pvdb[stationstr+'NAT_START' ] = {'type' : 'int',
                                     'count': NChans,
                                     'value' : [NChans]*NChans }
    pvdb[stationstr+'NAT_GATE' ] = {'type' : 'int',
                                    'count': NChans,
                                    'value' : [200]*NChans }
    pvdb[stationstr+'NAT_PS'   ] = {'type' : 'int',
                                    'count': NChans,
                                    'value' : [0]*NChans }

    pvdb[stationstr+'RESET'  ] = {'type' : 'int',
                                  'value' : 0 }
    pvdb[stationstr+'PGPLOOPBACK'  ] = {'type' : 'int',
                                        'value' : 0 }
    pvdb[stationstr+'PGPSKPINTVL'  ] = {'type' : 'int',
                                        'value' : 0xfff0 }
    pvdb[stationstr+'FULLEVT'      ] = {'type' : 'int',
                                        'value' : NChans }
    pvdb[stationstr+'FULLSIZE'     ] = {'type' : 'int',
                                        'value' : 3072 }
    pvdb[stationstr+'TESTPATTERN'  ] = {'type' : 'int',
                                        'value' : 0 } # -1: off 1: rect, 2: sawtooth
    pvdb[stationstr+'TRIGSHIFT'  ] = {'type' : 'int',
                                      'value' : 0 }
    pvdb[stationstr+'SYNCE'       ] = {'type' : 'int',
                                      'value' : 0 }
    pvdb[stationstr+'SYNCELO'     ] = {'type' : 'int',
#                                       'value' : 2050 }
#                                       'value' : 1600 }
                                       'value' : 5500-175 }
    pvdb[stationstr+'SYNCEHI'     ] = {'type' : 'int',
#                                       'value' : 2NChans00 }
#                                       'value' : 1950 }
                                       'value' : 5500+175 }
    pvdb[stationstr+'SYNCO'       ] = {'type' : 'int',
                                      'value' : 0 }
    pvdb[stationstr+'SYNCOLO'     ] = {'type' : 'int',
#                                       'value' : 11800 }
#                                       'value' : 11NChans00 }
                                       'value' : 15200-175 }
    pvdb[stationstr+'SYNCOHI'     ] = {'type' : 'int',
#                                       'value': 12200 }
#                                       'value' : 11750 }
                                       'value' : 15200+175 }
    pvdb[stationstr+'WRFIFOCNT'  ] = {'type' : 'int',
                                      'count' : NChans,
                                      'value' : [0]*NChans }
    pvdb[stationstr+'RDFIFOCNT'  ] = {'type' : 'int',
                                      'count' : NChans,
                                      'value' : [0]*NChans }

    # Status Monitoring

    # Timing link frames
    pvdb[stationstr+'TIMFRAMECNT'  ] = {'type' : 'int',
                                        'value' : 0 }
    # Timing link cycles paused (deadtime)
    pvdb[stationstr+'TIMPAUSECNT'  ] = {'type' : 'int',
                                        'value' : 0 }
    # Trigger counts (rate)
    pvdb[stationstr+'TRIGCNT'      ] = {'type' : 'int',
                                        'value' : 0 }
    # Trigger counts (total)
    pvdb[stationstr+'TRIGCNTSUM'   ] = {'type' : 'int',
                                        'value' : 0 }
    # Header read counts (total)
    pvdb[stationstr+'READCNTSUM'   ] = {'type' : 'int',
                                        'value' : 0 }
    # Start counts (total)
    pvdb[stationstr+'STARTCNTSUM'   ] = {'type' : 'int',
                                        'value' : 0 }
    # Queue counts (total)
    pvdb[stationstr+'QUEUECNTSUM'   ] = {'type' : 'int',
                                        'value' : 0 }
    # Msg Delay
    pvdb[stationstr+'MSGDELAYSET'   ] = {'type' : 'int',
                                        'value' : 0 }
    # Msg Delay
    pvdb[stationstr+'MSGDELAYGET'   ] = {'type' : 'int',
                                        'value' : 0 }
    # Header Count
    pvdb[stationstr+'HEADERCNTL0'   ] = {'type' : 'int',
                                         'value' : 0 }
    # Header Count
    pvdb[stationstr+'HEADERCNTOF'   ] = {'type' : 'int',
                                         'value' : 0 }
    # PGP link status
    pvdb[stationstr+'PGPLOCLINKRDY'] = {'type' : 'int',
                                        'count': NLanes,
                                        'value' : [0]*NLanes }
    pvdb[stationstr+'PGPREMLINKRDY'] = {'type' : 'int',
                                        'count': NLanes,
                                        'value' : [0]*NLanes }
    # PGP reference clocks
    pvdb[stationstr+'PGPTXCLKFREQ' ] = {'type' : 'int',
                                        'count': NLanes,
                                        'value' : [0]*NLanes }
    pvdb[stationstr+'PGPRXCLKFREQ' ] = {'type' : 'int',
                                        'count': NLanes,
                                        'value' : [0]*NLanes }
    # PGP frames transmitted
    pvdb[stationstr+'PGPTXCNT'     ] = {'type' : 'int',
                                        'count': NLanes,
                                        'value' : [0]*NLanes }
    pvdb[stationstr+'PGPTXCNTSUM'  ] = {'type' : 'int',
                                        'count': NLanes,
                                        'value' : [0]*NLanes }
    # PGP frames error in transmission
    pvdb[stationstr+'PGPTXERRCNT'  ] = {'type' : 'int',
                                        'count': NLanes,
                                        'value' : [0]*NLanes }
    # PGP frames received (deadtime link)
    pvdb[stationstr+'PGPRXCNT'     ] = {'type' : 'int',
                                        'count': NLanes,
                                        'value' : [0]*NLanes }
    # Last PGP opcode received
    pvdb[stationstr+'PGPRXLAST'    ] = {'type' : 'int',
                                        'count': NLanes,
                                        'value' : [0]*NLanes }
    # Last PGP opcode received
    pvdb[stationstr+'PGPREMPAUSE'  ] = {'type' : 'int',
                                        'count': NLanes,
                                        'value' : [0]*NLanes }

    # Bytes(?) free in buffer pool
    pvdb[stationstr+'RAW_FREEBUFSZ'] = {'type' : 'int',
                                        'count': NChans,
                                        'value' : [0]*NChans }
    # Events free in buffer pool
    pvdb[stationstr+'RAW_FREEBUFEVT'] = {'type' : 'int',
                                        'count': NChans,
                                        'value' : [0]*NChans }
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
                                        'count': NChans,
                                        'value' : [0]*NChans }
    # Events free in buffer pool
    pvdb[stationstr+'FEX_FREEBUFEVT'] = {'type' : 'int',
                                        'count': NChans,
                                        'value' : [0]*NChans }
    # Bytes(?) free in buffer pool
    pvdb[stationstr+'NAT_FREEBUFSZ'] = {'type' : 'int',
                                        'count': NChans,
                                        'value' : [0]*NChans }
    # Events free in buffer pool
    pvdb[stationstr+'NAT_FREEBUFEVT'] = {'type' : 'int',
                                        'count': NChans,
                                        'value' : [0]*NChans }

    # Data monitoring
    pvdb[stationstr+'RAWDATA'] = {'type' : 'int',
                                  'count': WfLen,
                                  'value' : [512]*WfLen }
    pvdb[stationstr+'FEXDATA'] = {'type' : 'int',
                                  'count': WfLen,
                                  'value' : [512]*WfLen }

    # Environmental monitoring
    pvdb[stationstr+'LOCAL12V'  ] = {'type'  : 'float',
                                     'value' : 0 }
    pvdb[stationstr+'EDGE12V'   ] = {'type'  : 'float',
                                     'value' : 0 }
    pvdb[stationstr+'AUX12V'    ] = {'type'  : 'float',
                                     'value' : 0 }
    pvdb[stationstr+'FMC12V'    ] = {'type'  : 'float',
                                     'value' : 0 }
    pvdb[stationstr+'BOARDTEMP' ] = {'type'  : 'float',
                                     'value' : 0 }
    pvdb[stationstr+'LOCAL3_3V' ] = {'type'  : 'float',
                                     'value' : 0 }
    pvdb[stationstr+'LOCAL2_5V' ] = {'type'  : 'float',
                                     'value' : 0 }
    pvdb[stationstr+'LOCAL1_8V' ] = {'type'  : 'float',
                                     'value' : 0 }
    pvdb[stationstr+'TOTALPOWER'] = {'type'  : 'float',
                                     'value' : 0 }
    pvdb[stationstr+'FMCPOWER'  ] = {'type'  : 'float',
                                     'value' : 0 }

    # printDb(pvdb, prefix)
    printDb()

    server = PVAServer(__name__)
    server.createPV(prefix, pvdb)

    # Save PVs to config dbase
    from pymongo import MongoClient, errors, ASCENDING, DESCENDING
    username = 'yoon82'
    host = 'psdb-dev'
    port = 9306
    daq_id = 'lcls2-tmo'
    dettype = 'hsd_cfg_2_4_3'
    client = MongoClient('mongodb://%s:%s@%s:%s' % (username, username, host, port))
    db = client[daq_id]
    collection = db[dettype]
    max_id = -1
    if daq_id not in client.database_names():
        print("Creating unique index")
        collection.create_index([('cfg_id', ASCENDING)], unique=True)
    else:
        max_id = collection.find_one(sort=[("cfg_id", DESCENDING)])["cfg_id"]
    _pvdb = pvdb
    _pvdb['cfg_id'] = max_id + 1 # this may be detector serial number
    print("#### cfg_id: ", max_id + 1)
    try:
        collection.insert(_pvdb)
    except errors.DuplicateKeyError:
        print("ID already exists. Exit without writing to database.")

    try:
        # process PVA transactions
        server.forever()
    except KeyboardInterrupt:
        print('\nInterrupted')



if __name__ == '__main__':
    main()
