import sys
import logging

from psdaq.epicstools.PVAServer import PVAServer
import time
from datetime import datetime
import argparse
#import socket
#import json
import pdb

Lanes = 4
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
    parser.add_argument('-D', '--use_db' , action='store_true', help='use mongodb')
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
    pvdb[stationstr+'BASE:APPLYUNCONFIG'] = {'type' : 'int', 
                                             'value' : 0 }
    pvdb[stationstr+'BASE:READY'        ] = {'type' : 'int', 
                                             'value' : 0 }

    # Specific PVs
    pvdb[stationstr+'ENABLE'   ] = {'type' : 'int', 
                                    'count': NChans,
                                    'value' : [1]*NChans }
    pvdb[stationstr+'RAW_START' ] = {'type' : 'int', 
                                     'count': NChans,
                                     'value' : [4]*NChans }
    pvdb[stationstr+'RAW_GATE' ] = {'type' : 'int', 
                                    'count': NChans,
                                    'value' : [200]*NChans }
    pvdb[stationstr+'RAW_PS'   ] = {'type' : 'int', 
                                    'count': NChans,
                                    'value' : [1]*NChans }
    pvdb[stationstr+'FEX_START' ] = {'type' : 'int', 
                                     'count': NChans,
                                     'value' : [4]*NChans }
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
                                       'value' : 11250-250 }
    pvdb[stationstr+'SYNCEHI'     ] = {'type' : 'int', 
                                       'value' : 11250+250 }
    pvdb[stationstr+'SYNCO'       ] = {'type' : 'int', 
                                       'value' : 0 }
    pvdb[stationstr+'SYNCOLO'     ] = {'type' : 'int', 
                                       'value' : 1450-200 }
    pvdb[stationstr+'SYNCOHI'     ] = {'type' : 'int', 
                                       'value' : 1450+200 }
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
                                        'count': Lanes,
                                        'value' : [0]*Lanes }
    pvdb[stationstr+'PGPREMLINKRDY'] = {'type' : 'int',
                                        'count': Lanes,
                                        'value' : [0]*Lanes }
    # PGP reference clocks
    pvdb[stationstr+'PGPTXCLKFREQ' ] = {'type' : 'int',
                                        'count': Lanes,
                                        'value' : [0]*Lanes }
    pvdb[stationstr+'PGPRXCLKFREQ' ] = {'type' : 'int',
                                        'count': Lanes,
                                        'value' : [0]*Lanes }
    # PGP frames transmitted
    pvdb[stationstr+'PGPTXCNT'     ] = {'type' : 'int',
                                        'count': Lanes,
                                        'value' : [0]*Lanes }
    pvdb[stationstr+'PGPTXCNTSUM'  ] = {'type' : 'int',
                                        'count': Lanes,
                                        'value' : [0]*Lanes }
    # PGP frames error in transmission
    pvdb[stationstr+'PGPTXERRCNT'  ] = {'type' : 'int',
                                        'count': Lanes,
                                        'value' : [0]*Lanes }
    # PGP frames received (deadtime link)
    pvdb[stationstr+'PGPRXCNT'     ] = {'type' : 'int',
                                        'count': Lanes,
                                        'value' : [0]*Lanes }
    # Last PGP opcode received
    pvdb[stationstr+'PGPRXLAST'    ] = {'type' : 'int',
                                        'count': Lanes,
                                        'value' : [0]*Lanes }
    # Last PGP opcode received
    pvdb[stationstr+'PGPREMPAUSE'  ] = {'type' : 'int',
                                        'count': Lanes,
                                        'value' : [0]*Lanes }

    # FIFO Overflows
    pvdb[stationstr+'DATA_FIFOOF']   = {'type' : 'int',
                                        'count': NChans,
                                        'value' : [0]*NChans }

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
    # FIFO Overflows
    pvdb[stationstr+'RAW_FIFOOF']        = {'type' : 'int',
                                            'count': NChans,
                                            'value' : [0]*NChans }
    # Bytes(?) free in buffer pool
    pvdb[stationstr+'FEX_FREEBUFSZ'] = {'type' : 'int',
                                        'count': NChans,
                                        'value' : [0]*NChans }
    # Events free in buffer pool
    pvdb[stationstr+'FEX_FREEBUFEVT'] = {'type' : 'int',
                                         'count': NChans,
                                         'value' : [0]*NChans }
    # FIFO Overflows
    pvdb[stationstr+'FEX_FIFOOF']        = {'type' : 'int',
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

    if args.use_db:
        # Save PVs to config dbase
        from pymongo import MongoClient, errors
        username = 'yoon82'
        host = 'psdb-dev'
        port = 9306
        instrument = 'amo'
        client = MongoClient('mongodb://%s:%s@%s:%s' % (username, username, host, port))
        db = client['config_db']
        collection = db[instrument]
        _pvdb = pvdb
        _pvdb['_id'] = 234 # this may be detector serial number
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
