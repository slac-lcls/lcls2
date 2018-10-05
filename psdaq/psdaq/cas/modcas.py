import sys

from pcaspy import SimpleServer, Driver
import time
from datetime import datetime
import argparse
#import socket
#import json
import pdb

NDsLinks    = 7
NAmcs       = 2
NPartitions = 16

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

def main():
    global pvdb
    pvdb = {}     # start with empty dictionary
    global prefix
    prefix = ''

    parser = argparse.ArgumentParser(prog=sys.argv[0], description='host PVs for XPM')

    parser.add_argument('-P', required=True, help='e.g. DAQ:LAB2:XPM:1', metavar='PREFIX')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')

    args = parser.parse_args()
    myDriver.verbose = args.verbose

    prefix = args.P
    
    # PVs
#    pvdb[':PARTITIONS'         ] = {'type' : 'int', 'value' : 255}
    pvdb[':PAddr'              ] = {'type' : 'int'}
    pvdb[':FwBuild'            ] = {'type' : 'char', 'count':256}
    pvdb[':ModuleInit'         ] = {'type' : 'int'}
    for i in range(NAmcs):
        pvdb[':DumpPll' + '%d'%i] = {'type' : 'int'}

    for i in range(2):
        pvdb[':DumpTiming%d'%i ] = {'type' : 'int'}

    pvdb[':Inhibit'            ] = {'type' : 'int'}
    pvdb[':TagStream'          ] = {'type' : 'int'}

    LinkEnable = [0]*32
    LinkEnable[17:19] = [1]*3  # DTIs in slots 3-5
    LinkEnable[4] = 1   # HSD on dev03
    LinkEnable[7] = 1   # HSD on dev02
    print(LinkEnable)

    for i in range(32):
        pvdb[':LinkTxDelay'  +'%d'%i] = {'type' : 'int'}
        pvdb[':LinkPartition'+'%d'%i] = {'type' : 'int'}
        pvdb[':LinkTrgSrc'   +'%d'%i] = {'type' : 'int'}
        pvdb[':LinkLoopback' +'%d'%i] = {'type' : 'int'}
        pvdb[':TxLinkReset'  +'%d'%i] = {'type' : 'int'}
        pvdb[':RxLinkReset'  +'%d'%i] = {'type' : 'int'}
        pvdb[':RxLinkDump'   +'%d'%i] = {'type' : 'int'}
        pvdb[':LinkEnable'   +'%d'%i] = {'type' : 'int', 'value' : LinkEnable[i] }
        pvdb[':LinkTxReady'  +'%d'%i] = {'type' : 'int'}
        pvdb[':LinkRxReady'  +'%d'%i] = {'type' : 'int'}
        pvdb[':LinkTxResetDone'  +'%d'%i] = {'type' : 'int'}
        pvdb[':LinkRxResetDone'  +'%d'%i] = {'type' : 'int'}
        pvdb[':LinkRxRcv'    +'%d'%i] = {'type' : 'int'}
        pvdb[':LinkRxErr'    +'%d'%i] = {'type' : 'int'}
        pvdb[':LinkIsXpm'    +'%d'%i] = {'type' : 'int'}
        pvdb[':RemoteLinkId'  +'%d'%i] = {'type' : 'int'}

    for i in range(14):
        pvdb[':LinkLabel'    +'%d'%i] = {'type' : 'string', 'value' : 'FP-%d'%i}

    for i in range(16,21):
        pvdb[':LinkLabel'    +'%d'%i] = {'type' : 'string', 'value' : 'BP-%d'%(i-13)}

    pvdb[':LinkId'] = {'type' : 'int', 'count' : 22}

    for i in range(NAmcs):
        pvdb[':PLL_LOS'       +'%d'%i] = {'type' : 'int'}
        pvdb[':PLL_LOL'       +'%d'%i] = {'type' : 'int'}
        pvdb[':PLL_BW_Select' +'%d'%i] = {'type' : 'int', 'value': 7}
        pvdb[':PLL_FreqTable' +'%d'%i] = {'type' : 'int', 'value': 2}
        pvdb[':PLL_FreqSelect'+'%d'%i] = {'type' : 'int', 'value': 89}
        pvdb[':PLL_Rate'      +'%d'%i] = {'type' : 'int', 'value': 10}
        pvdb[':PLL_PhaseInc'  +'%d'%i] = {'type' : 'int'}
        pvdb[':PLL_PhaseDec'  +'%d'%i] = {'type' : 'int'}
        pvdb[':PLL_Bypass'    +'%d'%i] = {'type' : 'int'}
        pvdb[':PLL_Reset'     +'%d'%i] = {'type' : 'int'}
        pvdb[':PLL_Skew'      +'%d'%i] = {'type' : 'int'}

    pvdb[':RxClks'     ] = {'type' : 'float', 'value': 0}
    pvdb[':TxClks'     ] = {'type' : 'float', 'value': 0}
    pvdb[':RxRsts'     ] = {'type' : 'float', 'value': 0}
    pvdb[':CrcErrs'    ] = {'type' : 'float', 'value': 0}
    pvdb[':RxDecErrs'  ] = {'type' : 'float', 'value': 0}
    pvdb[':RxDspErrs'  ] = {'type' : 'float', 'value': 0}
    pvdb[':BypassRsts' ] = {'type' : 'float', 'value': 0}
    pvdb[':BypassDones'] = {'type' : 'float', 'value': 0}
    pvdb[':RxLinkUp'   ] = {'type' : 'float', 'value': 0}
    pvdb[':FIDs'       ] = {'type' : 'float', 'value': 0}
    pvdb[':SOFs'       ] = {'type' : 'float', 'value': 0}
    pvdb[':EOFs'       ] = {'type' : 'float', 'value': 0}

    pvdb[':BpClk'      ] = {'type' : 'float', 'value': 0}

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

if __name__ == '__main__':
    main()
