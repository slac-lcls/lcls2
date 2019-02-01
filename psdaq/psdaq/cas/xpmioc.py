import sys
import socket
import argparse
import time
from psdaq.cas.pvedit import *
#from p4p.client.thread import Context
import logging

NPartitions = 8

stats = [{'name' : 'RxClks'     , 'value' : 185.7     , 'delta' : 0.01},
         {'name' : 'TxClks'     , 'value' : 185.7     , 'delta' : 0.01},
         {'name' : 'RxRsts'     , 'value' : 0         , 'delta' : 0},
         {'name' : 'CrcErrs'    , 'value' : 0         , 'delta' : 0},
         {'name' : 'RxDecErrs'  , 'value' : 0         , 'delta' : 0},
         {'name' : 'RxDspErrs'  , 'value' : 0         , 'delta' : 0},
         {'name' : 'BypassRsts' , 'value' : 0         , 'delta' : 0},
         {'name' : 'BypassDones', 'value' : 0         , 'delta' : 0},
         {'name' : 'RxLinkUp'   , 'value' : 1         , 'delta' : 0},
         {'name' : 'FIDs'       , 'value' : 1.3e6/1.4 , 'delta' : 1.e2},
         {'name' : 'SOFs'       , 'value' : 1.3e6/1.4 , 'delta' : 1.e2},
         {'name' : 'EOFs'       , 'value' : 1.3e6/1.4 , 'delta' : 1.e2}]
lstats = [{'name':'LinkTxReady'     , 'value' : 1, 'delta' : 0},
          {'name':'LinkRxReady'     , 'value' : 1, 'delta' : 0},
          {'name':'LinkTxResetDone' , 'value' : 1, 'delta' : 0},
          {'name':'LinkRxResetDone' , 'value' : 1, 'delta' : 0},
          {'name':'LinkRxRcv'       , 'value' : 8.5e6, 'delta' : 0.01e6},
          {'name':'LinkRxErr'       , 'value' : 0, 'delta' : 0},
          {'name':'LinkIsXpm'       , 'value' : 0, 'delta' : 0},
          {'name':'RemoteLinkId'    , 'value' : 0, 'delta' : 0}]

class PVStats:
    def __init__(self,pvbase):
        self.pvs = []
        self.ncall = 0
        for s in stats:
            self.pvs.append(Pv(pvbase+':'+s['name']))
        self.lpvs = []
        for i in range(32):
            lpv = []
            for s in lstats:
                lpv.append(Pv(pvbase+':'+s['name']+'%d'%i))
            self.lpvs.append(lpv)

    def update(self):
        for i,s in enumerate(stats):
            self.pvs[i].put(s['value']+s['delta']*self.ncall)
        for i in range(len(self.lpvs)):
            for j,s in enumerate(lstats):
                self.lpvs[i][j].put(s['value']+s['delta']*self.ncall)
        if self.ncall==5:
            self.ncall=0
        else:
            self.ncall+=1

class PVCtrls:
    def __init__(self,pvbase):
        pass

    def update(self):
        pass

class PVP:
    def __init__(self,pvbase,i):
        pass
        
    def update(self):
        pass

def main():

    parser = argparse.ArgumentParser(prog=sys.argv[0], description='Python simulation for XPM')

    parser.add_argument('-P', required=True, help='e.g. DAQ:LAB2:XPM:2', metavar='PREFIX')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')

    args = parser.parse_args()
    pvbase = args.P
    ppvbase = pvbase.rsplit(':',2)[0]
    print(pvbase,ppvbase)

    paddr   = Pv(pvbase+':PAddr')
    paddr.put(0xffffffff)

    fwBuild = Pv(pvbase+':FwBuild')
    fwBuild.put('xpm Python P4P Simulation')

    pvs = PVStats(pvbase)
    pvc = PVCtrls(pvbase)

    pvp = []
    for i in range(NPartitions):
        pvp.append(PVP(ppvbase,i))
    
    while True:
        time.sleep(1)

        pvs.update()
        
        for i in range(NPartitions):
            pvp[i].update()
    
