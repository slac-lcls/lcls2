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

class NoLock(object):
    def __init__(self):
        self._level=0
        self._lock = Lock()

    def acquire(self):
        if self._level!=0:
            print('NoLock.acquire level {}'.format(self._level))
        self._level=self._level+1

    def release(self):
        if self._level!=1:
            print('NoLock.release level {}'.format(self._level))
        self._level=self._level-1

def main():
    global pvdb
    pvdb = {}     # start with empty dictionary
    global prefix
    prefix = ''

    parser = argparse.ArgumentParser(prog=sys.argv[0], description='host PVs for XPM')

    parser.add_argument('-P', required=True, help='e.g. DAQ:LAB2:XPM:1', metavar='PREFIX')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')
    parser.add_argument('--ip', type=str, required=True, help="IP address" )
    parser.add_argument('--db', type=str, default=None, help="save/restore db, for example [https://pswww.slac.stanford.edu/ws-auth/devconfigdb/ws/,configDB,LAB2,PROD]")
    parser.add_argument('-I', action='store_true', help='initialize Cu timing')
    parser.add_argument('-L', action='store_true', help='bypass AMC Locks')
    parser.add_argument('-T', action='store_true', help='test mode')
    parser.add_argument('-F', type=float, default=1.076923e-6, help='fiducial period (sec)')
    parser.add_argument('-C', type=int, default=200, help='clocks per fiducial')

    args = parser.parse_args()
    if args.verbose:
#        logging.basicConfig(level=logging.DEBUG)
        setVerbose(True)

    # Set base
    base = pr.Root(name='AMCc',description='') 

    base.add(Top(
        name   = 'XPM',
        ipAddr = args.ip,
        fidPrescale = args.C,
    ))
    
    # Start the system
    base.start(
#        pollEn   = False,
#        initRead = False,
#        zmqPort  = None,
    )

    xpm = base.XPM
    app = base.XPM.XpmApp
    axiv = base.XPM.AxiVersion

    # Print the AxiVersion Summary
    axiv.printStatus()

    provider = StaticProvider(__name__)

    lock = Lock()

    pvstats = PVStats(provider, lock, args.P, xpm, args.F, axiv)
#    pvctrls = PVCtrls(provider, lock, name=args.P, ip=args.ip, xpm=xpm, stats=pvstats._groups, handle=pvstats.handle, db=args.db, cuInit=True)
    pvctrls = PVCtrls(provider, lock, name=args.P, ip=args.ip, xpm=xpm, stats=pvstats._groups, handle=pvstats.handle, db=args.db, cuInit=args.I, fidPrescale=args.C, fidPeriod=args.F*1.e9)

    cuMode='xtpg' in xpm.AxiVersion.ImageName.get()
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
                pvstats.update(cycle,cuMode)
                pvctrls.update(cycle)
                #  We have to delay the startup of some classes
                if cycle == 5:
                    pvxtpg  = PVXTpg(provider, lock, args.P, xpm, xpm.mmcmParms, cuMode, bypassLock=args.L)
                    pvxtpg.init()
                elif cycle < 5:
                    print('pvxtpg in %d'%(5-cycle))
                if pvxtpg is not None:
                    pvxtpg .update()
                curr  = time.perf_counter()
                delta = prev+updatePeriod-curr
#                print('Delta {:.2f}  Update {:.2f}  curr {:.2f}  prev {:.2f}'.format(delta,curr-prev,curr,prev))
                if delta>0:
                    time.sleep(delta)
                cycle += 1
        except KeyboardInterrupt:
            pass

if __name__ == '__main__':
    main()
