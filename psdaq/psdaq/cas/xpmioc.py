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

    def expired(self):
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

fixedRate = [1.3e6/1.4, 1.e6/14., 1.e6/98., 1.e6/980., 1.e6/9800, 1.e6/98000, 1.e6/980000]

class PVP:
    def __init__(self,pvbase):
        self.l0Select  = Pv(pvbase+'L0Select'          ,self.update)
        self.l0SelectF = Pv(pvbase+'L0Select_FixedRate',self.update)
        self.resetL0   = Pv(pvbase+'ResetL0'           ,self.update)
        self.run       = Pv(pvbase+'Run'               ,self.update)
        self.msgConfig = Pv(pvbase+'MsgConfig'         ,self.update)

        self.l0InpRate = Pv(pvbase+'L0InpRate')
        self.l0AccRate = Pv(pvbase+'L0AccRate')
        self.l1Rate    = Pv(pvbase+'L1Rate')
        self.runTime   = Pv(pvbase+'RunTime')
        self.numL0Inp  = Pv(pvbase+'NumL0Inp')
        self.numL0Acc  = Pv(pvbase+'NumL0Acc')
        self.numL1     = Pv(pvbase+'NumL1')
        self.deadFrac  = Pv(pvbase+'DeadFrac')
        self.deadTime  = Pv(pvbase+'DeadTime')

        self.enabled    = False
        self.reset()

    def reset (self):
        self.runTimeSec = 0
        self.numL0InpN  = 0
        self.numL0AccN  = 0
        self.numL1N     = 0

    def update(self,err=None):
        if self.resetL0.__value__:
            self.reset()

    def expired(self):
        r = 0
        if self.run.get():
            self.runTimeSec += 1
            if self.l0Select.get()==0:
                r = fixedRate[self.l0SelectF.get()]

        self.numL0InpN += r
        self.numL0AccN += r

        self.runTime  .put(self.runTimeSec)
        self.l0InpRate.put(r)
        self.l0AccRate.put(r)
        self.numL0Inp.put(self.numL0InpN)
        self.numL0Acc.put(self.numL0AccN)
        self.l1Rate  .put(0)
        self.numL1   .put(self.numL1N)
        self.deadFrac.put(0)
        self.deadTime.put(0)


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
        pvp.append(PVP(ppvbase+':PART:%d:'%i))
    
    while True:
        time.sleep(1)

        pvs.expired()
        
        for i in range(NPartitions):
            pvp[i].expired()
    
