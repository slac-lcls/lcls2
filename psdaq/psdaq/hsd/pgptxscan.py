import sys
import socket
import argparse
from threading import Thread, Event, Condition, Timer
import logging
import time

import rogue
import rogue.hardware.axi

import lcls2_pgp_pcie_apps
import surf.protocols.pgp      as pgp

import pyrogue as pr

from p4p.client.thread import Context

class Top(pr.Device):
    def __init__(self,
                 name   = 'KCU',
                 description = '',
                 memBase     = 0,
                 **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        for i in range(4):
            self.add(pgp.Pgp3AxiL(
                name            = (f'PgpMon[{i}]'),
                memBase         = memBase,
                offset          = (i*0x00010000)+0x00a08000,
                numVc           = 4,
                statusCountBits = 12,
                errorCountBits  = 8,
                writeEn         = True,
            ))


class ScanControl(object):

    def __init__(self,args):

        self.args = args

        self.base = pr.Root(name='DrpPgpIlv',description='')
        coreMap = rogue.hardware.axi.AxiMemMap(args.dev)
        self.base.add(Top(memBase = coreMap))

        self.rxCnt = [getattr(self.base.KCU,f'PgpMon[{i}]').RxFrameCount      for i in range(4)]
        self.rxErr = [getattr(self.base.KCU,f'PgpMon[{i}]').RxFrameErrorCount for i in range(4)]

        self.base.start()


        self.step_done = Event()

    def expired(self):
        self.step_done.set()

    def run(self):
        for i in range(self.args.s):
            logging.info(f'begin step {i}')
            rxCntStart  = [self.rxCnt[i].get() for i in range(4)]
            rxErrStart  = [self.rxErr[i].get() for i in range(4)]
            self.step_done.clear()
            self.transitions = Timer(self.args.t,self.expired)
            self.transitions.start()
            self.step_done.wait()
            logging.info(f'end step {i}')
            rxCntStop  = [self.rxCnt[i].get() for i in range(4)]
            rxErrStop  = [self.rxErr[i].get() for i in range(4)]
            rxCntDiff  = [rxCntStop[i]-rxCntStart[i] for i in range(4)]
            errorFrac  = [(rxErrStop[i]-rxErrStart[i])/rxCntDiff[i] if rxCntDiff[i]>0 else -1. for i in range(4)]
            print(f'errFrac {errorFrac}')

def main():

    parser = argparse.ArgumentParser(description='hsd pgp tx scan test')
    parser.add_argument('-t', metavar='TIMER', type=float, default=1.0, help='timer')
    parser.add_argument('-n', metavar='NCYCLE', type=int, default=1, help='ncycles')
    parser.add_argument('-s', metavar='STEPS', type=int, default=5, help='nsteps')
    parser.add_argument('--dev', metavar='DEV', default='/dev/datadev_1', help='pgp device file')

    args = parser.parse_args()
    
    #logging.basicConfig(level=logging.DEBUG)

    ctxt = Context('pva')
    cfgname = 'DAQ:RIX:HSD:1_1B:B:PGPCONFIG'
    cfg  = ctxt.get(cfgname)
    print(f'cfg {cfg}')

    c = ScanControl(args)
    for prec in range(1,15,2):
        cfg['precursor'] = [prec]*4
        for posc in range(1,15,2):
            cfg['postcursor'] = [posc]*4
            ctxt.put(cfgname,cfg,wait=True)
            time.sleep(0.01)
            print(f'precursor {prec}  postcursor {posc}')
            c.run()

    c.base.stop()
    print('Done')

if __name__ == '__main__':
    main()
