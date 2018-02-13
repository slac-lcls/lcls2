import sys

from pcaspy import SimpleServer, Driver
import time
from datetime import datetime
import argparse
import pdb

# yaml metadata
numUsLinks = 7


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

    parser = argparse.ArgumentParser(prog=sys.argv[0], description='host PVs for DTI(s)')

    parser.add_argument('-P', required=True, metavar='PREFIX', help='common prefix, e.g. DAQ')
    parser.add_argument('-R', metavar='P1[,P2[...]]', help='unique partition(s)')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')

    args = parser.parse_args()
    myDriver.verbose = args.verbose

    prefix = args.P+':'

    if args.R is None:
      p2set = set([''])
    else:
      # add colon to each item
      p2set = set([s+':' for s in args.R.split(",")])

    # PVs

    for p2 in p2set:
      for i in range (numUsLinks):
        pvdb[p2+'UsLinkEn'        +'%d'%i] = {'type' : 'int'}
        pvdb[p2+'UsLinkTagEn'     +'%d'%i] = {'type' : 'int'}
        pvdb[p2+'UsLinkL1En'      +'%d'%i] = {'type' : 'int'}
        pvdb[p2+'UsLinkPartition' +'%d'%i] = {'type' : 'int'}
        pvdb[p2+'UsLinkTrigDelay' +'%d'%i] = {'type' : 'int'}
        pvdb[p2+'UsLinkFwdMask'   +'%d'%i] = {'type' : 'int'}
        pvdb[p2+'UsLinkFwdMode'   +'%d'%i] = {'type' : 'int'}
        pvdb[p2+'UsLinkDataSrc'   +'%d'%i] = {'type' : 'int'}
        pvdb[p2+'UsLinkDataType'  +'%d'%i] = {'type' : 'int'}
        pvdb[p2+'UsLinkLabel'     +'%d'%i] = {'type' : 'string', 'value' : 'US-%d'%i }

      pvdb[p2+'ModuleInit'     ] = {'type' : 'int'}

      pvdb[p2+'CountClear'     ] = {'type' : 'int'}
      pvdb[p2+'CountUpdate'    ] = {'type' : 'int'}
      pvdb[p2+'UsLinkUp'       ] = {'type' : 'int'}
      pvdb[p2+'BpLinkUp'       ] = {'type' : 'int'}
      pvdb[p2+'DsLinkUp'       ] = {'type' : 'int'}

      pvdb[p2+'UsRxErrs'       ] = {'type' : 'int'}
      pvdb[p2+'dUsRxErrs'      ] = {'type' : 'float'}
      pvdb[p2+'UsRemLinkID'    ] = {'type' : 'int'}
      pvdb[p2+'UsRxFull'       ] = {'type' : 'int'}
      pvdb[p2+'dUsRxFull'      ] = {'type' : 'float'}
      pvdb[p2+'UsIbRecv'       ] = {'type' : 'int'}
      pvdb[p2+'dUsIbRecv'      ] = {'type' : 'float'}
      pvdb[p2+'UsIbDump'       ] = {'type' : 'int'}
      pvdb[p2+'UsIbEvt'        ] = {'type' : 'int'}
      pvdb[p2+'dUsIbEvt'       ] = {'type' : 'float'}
      pvdb[p2+'UsAppObRecv'    ] = {'type' : 'int'}
      pvdb[p2+'dUsAppObRecv'   ] = {'type' : 'float'}
      pvdb[p2+'UsAppObSent'    ] = {'type' : 'int'}
      pvdb[p2+'dUsAppObSent'   ] = {'type' : 'float'}
      pvdb[p2+'DsRxErrs'       ] = {'type' : 'int'}
      pvdb[p2+'dDsRxErrs'      ] = {'type' : 'float'}
      pvdb[p2+'DsRemLinkID'    ] = {'type' : 'int'}
      pvdb[p2+'DsRxFull'       ] = {'type' : 'int'}
      pvdb[p2+'dDsRxFull'      ] = {'type' : 'float'}
      pvdb[p2+'DsObSent'       ] = {'type' : 'int'}
      pvdb[p2+'dDsObSent'      ] = {'type' : 'float'}

      pvdb[p2+'QpllLock'       ] = {'type' : 'int'}
      pvdb[p2+'BpTxInterval'   ] = {'type' : 'int'}

      pvdb[p2+'MonClkRate'     ] = {'type' : 'int', 'count' : 4}
      pvdb[p2+'MonClkSlow'     ] = {'type' : 'int', 'count' : 4}
      pvdb[p2+'MonClkFast'     ] = {'type' : 'int', 'count' : 4}
      pvdb[p2+'MonClkLock'     ] = {'type' : 'int', 'count' : 4}

      pvdb[p2+'UsLinkObL0'     ] = {'type' : 'int'}
      pvdb[p2+'dUsLinkObL0'    ] = {'type' : 'float'}
      pvdb[p2+'UsLinkObL1A'    ] = {'type' : 'int'}
      pvdb[p2+'dUsLinkObL1A'   ] = {'type' : 'float'}
      pvdb[p2+'UsLinkObL1R'    ] = {'type' : 'int'}
      pvdb[p2+'dUsLinkObL1R'   ] = {'type' : 'float'}

      # The following PVs correspond to DtiDsPgp5Gb.yaml.
      pvdb[p2+'CountReset0'   ] = {'type' : 'int'}
      pvdb[p2+'CountReset1'   ] = {'type' : 'int'}

      pvdb[p2+'ResetRx'       ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'Flush'         ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'Loopback'      ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'TxLocData'     ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'TxLocDataEn'   ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'AutoStatSendEn'] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'FlowControlDis'] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'RxPhyRdy'      ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'TxPhyRdy'      ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'LocLinkRdy'    ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'RemLinkRdy'    ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'TxRdy'         ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'RxPolarity'    ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'RemPauseStat'  ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'LocPauseStat'  ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'RemOflowStat'  ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'LocOflowStat'  ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'RemData'       ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'CellErrs'      ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'LinkDowns'     ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'LinkErrs'      ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'RemOflowVC'    ] = {'type' : 'int', 'count' : 2 * 4}
      pvdb[p2+'RxFrErrs'      ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'dRxFrErrs'     ] = {'type' : 'float', 'count' : 2}
      pvdb[p2+'RxFrames'      ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'dRxFrames'     ] = {'type' : 'float', 'count' : 2}
      pvdb[p2+'LocOflowVC'    ] = {'type' : 'int', 'count' : 2 * 4}
      pvdb[p2+'TxFrErrs'      ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'dTxFrErrs'     ] = {'type' : 'float', 'count' : 2}
      pvdb[p2+'TxFrames'      ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'dTxFrames'     ] = {'type' : 'float', 'count' : 2}
      pvdb[p2+'RxClockFreq'   ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'TxClockFreq'   ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'TxLastOpCode'  ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'RxLastOpCode'  ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'RxOpCodes'     ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'dRxOpCodes'    ] = {'type' : 'float', 'count' : 2}
      pvdb[p2+'TxOpCodes'     ] = {'type' : 'int', 'count' : 2}
      pvdb[p2+'dTxOpCodes'    ] = {'type' : 'float', 'count' : 2}

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
