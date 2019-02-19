import sys
import logging

from psdaq.epicstools.PVAServer import PVAServer
import time
from datetime import datetime
import argparse
#import socket
#import json
import pdb

NPartitions = 8

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

    parser.add_argument('-P', required=True, help='DAQ:LAB2', metavar='PREFIX')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    stationstr = 'PART'
    prefix = args.P+':'

    # PVs

    for i in range(NPartitions):
        pvdb[stationstr+':%d:XPM'                %i] = {'type' : 'int', 'value': 2}
        pvdb[stationstr+':%d:L0Select'           %i] = {'type' : 'int'}
        pvdb[stationstr+':%d:L0Select_FixedRate' %i] = {'type' : 'int'}
        pvdb[stationstr+':%d:L0Select_ACRate'    %i] = {'type' : 'int'}
        pvdb[stationstr+':%d:L0Select_ACTimeslot'%i] = {'type' : 'int'}
        pvdb[stationstr+':%d:L0Select_Sequence'  %i] = {'type' : 'int', 'value': 16}
        pvdb[stationstr+':%d:L0Select_SeqBit'    %i] = {'type' : 'int'}
        pvdb[stationstr+':%d:DstSelect'          %i] = {'type' : 'int', 'value': 1}
        pvdb[stationstr+':%d:DstSelect_Mask'     %i] = {'type' : 'int'}
        pvdb[stationstr+':%d:L0Delay'            %i] = {'type' : 'int', 'value': 99}
        pvdb[stationstr+':%d:ResetL0'            %i] = {'type' : 'int'}
        pvdb[stationstr+':%d:Run'                %i] = {'type' : 'int'}

        pvdb[stationstr+':%d:L1TrgClear'   %i] = {'type' : 'int'}
        pvdb[stationstr+':%d:L1TrgEnable'  %i] = {'type' : 'int'}
        pvdb[stationstr+':%d:L1TrgSource'  %i] = {'type' : 'int'}
        pvdb[stationstr+':%d:L1TrgWord'    %i] = {'type' : 'int'}
        pvdb[stationstr+':%d:L1TrgWrite'   %i] = {'type' : 'int'}
        pvdb[stationstr+':%d:AnaTagReset'  %i] = {'type' : 'int'}
        pvdb[stationstr+':%d:AnaTag'       %i] = {'type' : 'int'}
        pvdb[stationstr+':%d:AnaTagPush'   %i] = {'type' : 'int'}
        pvdb[stationstr+':%d:AnaTagWrite'  %i] = {'type' : 'int'}
        pvdb[stationstr+':%d:PipelineDepth'%i] = {'type' : 'int'}
        #  Generic message interface
        pvdb[stationstr+':%d:MsgHeader'    %i] = {'type' : 'int'}
        pvdb[stationstr+':%d:MsgInsert'    %i] = {'type' : 'int', 'value': 0}
        pvdb[stationstr+':%d:MsgPayload'   %i] = {'type' : 'int'}
        #  Specific messages
        pvdb[stationstr+':%d:MsgConfig'    %i] = {'type' : 'int'}
        pvdb[stationstr+':%d:MsgConfigKey' %i] = {'type' : 'int'}
        pvdb[stationstr+':%d:MsgUnconfig'  %i] = {'type' : 'int'}
        pvdb[stationstr+':%d:MsgEnable'    %i] = {'type' : 'int'}
        pvdb[stationstr+':%d:MsgDisable'   %i] = {'type' : 'int'}
        pvdb[stationstr+':%d:MsgClear'     %i] = {'type' : 'int'}
        for j in range(4):
            pvdb[stationstr+':%d:InhInterval%d'  %(i,j)] = {'type' : 'int', 'value': 1}
            pvdb[stationstr+':%d:InhLimit%d'     %(i,j)] = {'type' : 'int', 'value': 1}
            pvdb[stationstr+':%d:InhEnable%d'    %(i,j)] = {'type' : 'int', 'value': 0}

        pvdb[stationstr+':%d:RunTime'  %i] = {'type' : 'float', 'value': 0}
        pvdb[stationstr+':%d:MsgDelay' %i] = {'type' : 'float', 'value': 0}
        pvdb[stationstr+':%d:L0InpRate'%i] = {'type' : 'float', 'value': 0, 'extra': [('MDEL', 'float', 0.1)]}
        pvdb[stationstr+':%d:L0AccRate'%i] = {'type' : 'float', 'value': 0}
        pvdb[stationstr+':%d:L1Rate'   %i] = {'type' : 'float', 'value': 0}
        pvdb[stationstr+':%d:NumL0Inp' %i] = {'type' : 'float', 'value': 0}
        pvdb[stationstr+':%d:NumL0Acc' %i] = {'type' : 'float', 'value': 0}
        pvdb[stationstr+':%d:NumL1'    %i] = {'type' : 'float', 'value': 0}
        pvdb[stationstr+':%d:DeadFrac' %i] = {'type' : 'float', 'value': 0}
        pvdb[stationstr+':%d:DeadTime' %i] = {'type' : 'float', 'value': 0}

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
