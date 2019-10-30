#!/usr/bin/env python

import sys
import pyrogue as pr
import argparse
import socket
import time
import logging

from p4p.nt import NTTable
from p4p.server.thread import SharedPV
from p4p.server import Server, StaticProvider

from .Top import *

provider = None

class DefaultPVHandler(object):

    def __init__(self):
        pass

    def put(self, pv, op):
        postedval = op.value()
        postedval['timeStamp.secondsPastEpoch'], postedval['timeStamp.nanoseconds'] = divmod(float(time.time_ns()), 1.0e9)
        pv.post(postedval)
        op.done()

def toTable(t):
    table = []
    for v in t.items():
        table.append((v[0],v[1][0][1:]))
        n = len(v[1][1])
    return table,n

def toDict(t):
    d = {}
    for v in t.items():
        d[v[0]] = v[1][1]
    return d

def toDictList(t,n):
    l = []
    for i in range(n):
        d = {}
        for v in t.items():
            d[v[0]] = v[1][1][i]
        l.append(d)
    return l

def addPVT(name,t):
    table,n = toTable(t)
    init    = toDictList(t,n)
    pv = SharedPV(initial=NTTable(table).wrap(init),
                  handler=DefaultPVHandler())
    provider.add(name,pv)
    return pv

pvdef = {'RxPwr'  : ('af',[0]*4),
         'TxBiasI': ('af',[0]*4),
         'FullTT' : ('ai',[0]*4),
         'nFullTT': ('ai',[0]*4)}


class PVStats(object):

    def __init__(self, name, kcu, hsd=False):
        self.kcu   = kcu
        self.pv    = addPVT(name+':MON',pvdef)
        self.value = toDict(pvdef)
        self.hsd   = hsd
    
    def init(self):
        pass

    def update(self):
        self.kcu.I2cBus.selectDevice('QSFP0')
        v = self.kcu.I2cBus.QSFP0.getRxPwr()
        for i in range(len(v)):
            self.value['RxPwr'][i] = v[i]
        v = self.kcu.I2cBus.QSFP0.getTxBiasI()
        for i in range(len(v)):
            self.value['TxBiasI'][i] = v[i]
        if self.hsd:
            pass
        else:
            v = self.kcu.TDetSemi.getRTT()
            for i in range(len(v)):
                self.value['FullTT' ][i] = v[i][0]
                self.value['nFullTT'][i] = v[i][1]

        value = self.pv.current()
        value['value'] = self.value
        value['timeStamp.secondsPastEpoch'], value['timeStamp.nanoseconds'] = divmod(float(time.time_ns()), 1.0e9)
        self.pv.post(value)

def main():
    global pvdb
    pvdb = {}     # start with empty dictionary
    global prefix
    prefix = ''
    global provider

    parser = argparse.ArgumentParser(prog=sys.argv[0], description='host PVs for KCU')
    parser.add_argument('-i','--interval',type=int ,help='PV update interval',default=10)
    parser.add_argument('-H','--hsd'     ,action='store_true',help='HSD node',default=False)
    args = parser.parse_args()

    # Set base
    base = pr.Root(name='KCUr',description='') 

    coreMap = rogue.hardware.axi.AxiMemMap('/dev/datadev_0')

    base.add(Top(memBase = coreMap))
    
    # Start the system
    base.start(
        pollEn   = False,
        initRead = False,
        zmqPort  = None,
    )

    kcu = base.KCU

    if args.hsd:
        kcu.I2cBus.selectDevice('QSFP0')
        print(kcu.I2cBus.QSFP0.getRxPwr())
    else:
        print(kcu.TDetTiming.getClkRates())
        print(kcu.TDetSemi.getRTT())

    provider = StaticProvider(__name__)

    pvstats = PVStats('DAQ:LAB2:'+socket.gethostname().replace('-','_').upper(),kcu,args.hsd)

    # process PVA transactions
    updatePeriod = args.interval
    with Server(providers=[provider]):
        try:
            pvstats.init()
            while True:
                prev = time.perf_counter()
                pvstats.update()
                curr  = time.perf_counter()
                delta = prev+updatePeriod-curr
#                print('Delta {:.2f}  Update {:.2f}  curr {:.2f}  prev {:.2f}'.format(delta,curr-prev,curr,prev))
                if delta>0:
                    time.sleep(delta)
        except KeyboardInterrupt:
            pass

if __name__ == '__main__':
    main()
