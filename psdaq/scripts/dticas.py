import sys

from pcaspy import SimpleServer, Driver
import time
from datetime import datetime
import thread
import subprocess
import argparse
#import socket
#import json
import pdb

# yaml metadata
numUsLinks = 7

streamSet = { 'Us', 'Ds' }

class myDriver(Driver):
    def __init__(self):
        super(myDriver, self).__init__()


def printDb():
    global pvdb
    global prefix

    print '=========== Serving %d PVs ==============' % len(pvdb)
    for key in sorted(pvdb):
        print prefix+key
    print '========================================='
    return

if __name__ == '__main__':
    global pvdb
    pvdb = {}     # start with empty dictionary
    global prefix
    prefix = ''

    parser = argparse.ArgumentParser(prog=sys.argv[0], description='host PVs for DTI')

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
    if (args.P).find(":") > 0:
        instrument, suffix = (args.P).split(':', 1)
        try:
            station = int(suffix)
        except:
            station = 0
        stationstr = str(station)
    else:
        instrument = args.P
        station = 0
        stationstr = ''

    # PVs

    for i in range (numUsLinks):
      pvdb[stationstr+':DTI:UsLinkEn'        +'%d'%i] = {'type' : 'int'}
      pvdb[stationstr+':DTI:UsLinkTagEn'     +'%d'%i] = {'type' : 'int'}
      pvdb[stationstr+':DTI:UsLinkL1En'      +'%d'%i] = {'type' : 'int'}
      pvdb[stationstr+':DTI:UsLinkPartition' +'%d'%i] = {'type' : 'int'}
      pvdb[stationstr+':DTI:UsLinkTrigDelay' +'%d'%i] = {'type' : 'int'}
      pvdb[stationstr+':DTI:UsLinkFwdMask'   +'%d'%i] = {'type' : 'int'}
      pvdb[stationstr+':DTI:UsLinkFwdMode'   +'%d'%i] = {'type' : 'int'}
      pvdb[stationstr+':DTI:UsLinkDataSrc'   +'%d'%i] = {'type' : 'int'}
      pvdb[stationstr+':DTI:UsLinkDataType'  +'%d'%i] = {'type' : 'int'}
      pvdb[stationstr+':DTI:UsLinkLabel'     +'%d'%i] = {'type' : 'string', 'value' : 'US-%d'%i }

    pvdb[stationstr+':DTI:ModuleInit'     ] = {'type' : 'int'}

    pvdb[stationstr+':DTI:CountClear'     ] = {'type' : 'int'}
    pvdb[stationstr+':DTI:CountUpdate'    ] = {'type' : 'int'}
    pvdb[stationstr+':DTI:UsLinkUp'       ] = {'type' : 'int'}
    pvdb[stationstr+':DTI:BpLinkUp'       ] = {'type' : 'int'}
    pvdb[stationstr+':DTI:DsLinkUp'       ] = {'type' : 'int'}

    pvdb[stationstr+':DTI:UsRxErrs'       ] = {'type' : 'int'}
    pvdb[stationstr+':DTI:dUsRxErrs'      ] = {'type' : 'float'}
    pvdb[stationstr+':DTI:UsRemLinkID'    ] = {'type' : 'int'}
    pvdb[stationstr+':DTI:UsRxFull'       ] = {'type' : 'int'}
    pvdb[stationstr+':DTI:dUsRxFull'      ] = {'type' : 'float'}
    pvdb[stationstr+':DTI:UsIbRecv'       ] = {'type' : 'int'}
    pvdb[stationstr+':DTI:dUsIbRecv'      ] = {'type' : 'float'}
    pvdb[stationstr+':DTI:UsIbDump'       ] = {'type' : 'int'}
    pvdb[stationstr+':DTI:UsIbEvt'        ] = {'type' : 'int'}
    pvdb[stationstr+':DTI:dUsIbEvt'       ] = {'type' : 'float'}
    pvdb[stationstr+':DTI:UsAppObRecv'    ] = {'type' : 'int'}
    pvdb[stationstr+':DTI:dUsAppObRecv'   ] = {'type' : 'float'}
    pvdb[stationstr+':DTI:UsAppObSent'    ] = {'type' : 'int'}
    pvdb[stationstr+':DTI:dUsAppObSent'   ] = {'type' : 'float'}
    pvdb[stationstr+':DTI:DsRxErrs'       ] = {'type' : 'int'}
    pvdb[stationstr+':DTI:dDsRxErrs'      ] = {'type' : 'float'}
    pvdb[stationstr+':DTI:DsRemLinkID'    ] = {'type' : 'int'}
    pvdb[stationstr+':DTI:DsRxFull'       ] = {'type' : 'int'}
    pvdb[stationstr+':DTI:dDsRxFull'      ] = {'type' : 'float'}
    pvdb[stationstr+':DTI:DsObSent'       ] = {'type' : 'int'}
    pvdb[stationstr+':DTI:dDsObSent'      ] = {'type' : 'float'}

    pvdb[stationstr+':DTI:QpllLock'       ] = {'type' : 'int'}
    pvdb[stationstr+':DTI:BpTxInterval'   ] = {'type' : 'int'}

    pvdb[stationstr+':DTI:MonClkRate'     ] = {'type' : 'int', 'count' : 4}
    pvdb[stationstr+':DTI:MonClkSlow'     ] = {'type' : 'int', 'count' : 4}
    pvdb[stationstr+':DTI:MonClkFast'     ] = {'type' : 'int', 'count' : 4}
    pvdb[stationstr+':DTI:MonClkLock'     ] = {'type' : 'int', 'count' : 4}

    pvdb[stationstr+':DTI:UsLinkObL0'     ] = {'type' : 'int'}
    pvdb[stationstr+':DTI:dUsLinkObL0'    ] = {'type' : 'float'}
    pvdb[stationstr+':DTI:UsLinkObL1A'    ] = {'type' : 'int'}
    pvdb[stationstr+':DTI:dUsLinkObL1A'   ] = {'type' : 'float'}
    pvdb[stationstr+':DTI:UsLinkObL1R'    ] = {'type' : 'int'}
    pvdb[stationstr+':DTI:dUsLinkObL1R'   ] = {'type' : 'float'}

    # The following PVs correspond to DtiDsPgp5Gb.yaml.
    pvdb[stationstr+':DTI:CountReset0'   ] = {'type' : 'int'}
    pvdb[stationstr+':DTI:CountReset1'   ] = {'type' : 'int'}

    pvdb[stationstr+':DTI:ResetRx'       ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:Flush'         ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:Loopback'      ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:TxLocData'     ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:TxLocDataEn'   ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:AutoStatSendEn'] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:FlowControlDis'] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:RxPhyRdy'      ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:TxPhyRdy'      ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:LocLinkRdy'    ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:RemLinkRdy'    ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:TxRdy'         ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:RxPolarity'    ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:RemPauseStat'  ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:LocPauseStat'  ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:RemOflowStat'  ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:LocOflowStat'  ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:RemData'       ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:CellErrs'      ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:LinkDowns'     ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:LinkErrs'      ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:RemOflowVC'    ] = {'type' : 'int', 'count' : 2 * 4}
    pvdb[stationstr+':DTI:RxFrErrs'      ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:dRxFrErrs'     ] = {'type' : 'float', 'count' : 2}
    pvdb[stationstr+':DTI:RxFrames'      ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:dRxFrames'     ] = {'type' : 'float', 'count' : 2}
    pvdb[stationstr+':DTI:LocOflowVC'    ] = {'type' : 'int', 'count' : 2 * 4}
    pvdb[stationstr+':DTI:TxFrErrs'      ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:dTxFrErrs'     ] = {'type' : 'float', 'count' : 2}
    pvdb[stationstr+':DTI:TxFrames'      ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:dTxFrames'     ] = {'type' : 'float', 'count' : 2}
    pvdb[stationstr+':DTI:RxClockFreq'   ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:TxClockFreq'   ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:TxLastOpCode'  ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:RxLastOpCode'  ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:RxOpCodes'     ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:dRxOpCodes'    ] = {'type' : 'float', 'count' : 2}
    pvdb[stationstr+':DTI:TxOpCodes'     ] = {'type' : 'int', 'count' : 2}
    pvdb[stationstr+':DTI:dTxOpCodes'    ] = {'type' : 'float', 'count' : 2}

    prefix = 'DAQ:' + instrument

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
        print '\nInterrupted'
