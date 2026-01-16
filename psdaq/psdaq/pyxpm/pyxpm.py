#!/usr/bin/env python

import sys
import pyrogue as pr
import argparse

import logging

from p4p.server import Server, StaticProvider
from threading import Lock

import time
from datetime import datetime
#import pdb

#import psdaq.pyxpm.xpm as xpmr
from psdaq.pyxpm.xpm.Top import *
from psdaq.pyxpm.pvstats import *
from psdaq.pyxpm.pvctrls import *
from psdaq.pyxpm.pvxtpg  import *
from psdaq.pyxpm.pvhandler import *
import psdaq.pyxpm.autosave as autosave

##MIN_FW_VERSION = 0x030c0100
#MIN_FW_VERSION = 0
MIN_FW_VERSION = 0x030d0400

class NoLock(object):
    def __init__(self):
        self._level=0
        self._lock = Lock()

    def acquire(self):
        if self._level!=0:
            logging.info('NoLock.acquire level {}'.format(self._level))
        self._level=self._level+1

    def release(self):
        if self._level!=1:
            logging.info('NoLock.release level {}'.format(self._level))
        self._level=self._level-1

class MyProvider(StaticProvider):
    def __init__(self, name):
        super(MyProvider,self).__init__(name)
        self.pvdict = {}

    def add(self,name,pv):
        self.pvdict[name] = pv
        super(MyProvider,self).add(name,pv)

def main():
    global pvdb
    pvdb = {}     # start with empty dictionary
    global prefix
    prefix = ''

    parser = argparse.ArgumentParser(prog=sys.argv[0], description='host PVs for XPM')

    parser.add_argument('-P', required=True, help='e.g. DAQ:LAB2:XPM:1', metavar='PREFIX')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')
    parser.add_argument('--ip', type=str, required=True, help="IP address" )
    parser.add_argument('--xvc', type=int, default=None, help="XVC port (e.g. 2542)" )
    parser.add_argument('--db', type=str, default=None, help="save/restore db, for example [https://pswww.slac.stanford.edu/ws-auth/devconfigdb/ws/,configDB,LAB2,PROD]")
    parser.add_argument('--norestore', action='store_true', help='skip restore (clean save)')
    parser.add_argument('-I', action='store_true', help='initialize Cu timing')
    parser.add_argument('-L', action='store_true', help='bypass AMC Locks')
    parser.add_argument('-T', action='store_true', help='test mode : use when no valid timing input')
    parser.add_argument('-F', type=float, default=1.076923e-6, help='fiducial period (sec)')
    parser.add_argument('-C', type=int, default=200, help='clocks per fiducial')
    parser.add_argument('-A', type=int, default=2, help='number of AMC cards')

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

    # Set base
    base = pr.Root(name='AMCc',description='',pollEn=False) 

    base.add(Top(
        name   = 'XPM',
        ipAddr = args.ip,
        xvcPort = args.xvc,
        fidPrescale = args.C,
        noTiming = args.T,
        fwVersion = MIN_FW_VERSION,
    ))
    
    # Start the system
    base.start()

    xpm = base.XPM
    xpm.start()

    app = base.XPM.XpmApp
    axiv = base.XPM.AxiVersion

    # Print the AxiVersion Summary
    axiv.printStatus()
    fwver = axiv.FpgaVersion.get()

    if fwver < MIN_FW_VERSION:
        raise RuntimeError(f'Firmware version {fwver:x} is less than min required {MIN_FW_VERSION:x}')

    #provider = StaticProvider(__name__)
    provider = MyProvider(__name__)

    lock = Lock()

    autosave.set(args.P,args.db,None,norestore=args.norestore)

    imageName = axiv.ImageName.get()
#    imageName = 'xpmGen'
    isXTPG = 'xtpg' in imageName
    isGen  = 'Gen' in imageName
    if isGen or isXTPG:
        xpm.TPGMini.setup(False)

    pvstats = PVStats(provider, lock, args.P, xpm, args.F, axiv, nAMCs=args.A, 
                      noTiming=args.T, fidRate=1./args.F)
    pvctrls = PVCtrls(provider, lock, name=args.P, ip=args.ip, xpm=xpm, stats=pvstats._groups, usTiming=pvstats._usTiming, handle=pvstats.handle, paddr=pvstats.paddr, db=args.db, cuInit=args.I, fidPrescale=args.C, fidPeriod=args.F*1.e9, imageName=imageName)

    pvxtpg = None

    # process PVA transactions
    updatePeriod = 1.0
    # test mode skips xtpg and pvseq
    cycle = 0 if not args.T else 20
    with Server(providers=[provider]):
        try:
            if pvxtpg is not None:
                pvxtpg .init()
            pvstats.init()
            while True:
                prev = time.perf_counter()
                pvstats.update(cycle,isGen,isXTPG)
                pvctrls.update(cycle)
                autosave.update()
                #  We have to delay the startup of some classes
                if cycle == 5 and isXTPG:
                    pvxtpg  = PVXTpg(provider, lock, args.P, xpm, xpm.mmcmParms, isXTPG, bypassLock=args.L)
                    pvxtpg.init()

                elif cycle == 10:   # Wait for PVSeq to register with autosave/restore
                    autosave.restore()

                elif cycle < 5:
                    logging.info('pvxtpg in %d'%(5-cycle))
                if pvxtpg is not None:
                    pvxtpg .update()
                curr  = time.perf_counter()
                delta = prev+updatePeriod-curr
#                logging.verbose('Delta {:.2f}  Update {:.2f}  curr {:.2f}  prev {:.2f}'.format(delta,curr-prev,curr,prev))
                if delta>0:
                    time.sleep(delta)
                cycle += 1
        except KeyboardInterrupt:
            pass

if __name__ == '__main__':
    main()
