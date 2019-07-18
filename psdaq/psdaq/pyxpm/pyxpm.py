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

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Set base
    base = pr.Root(name='AMCc',description='') 

    base.add(Top(
        name   = 'XPM',
        ipAddr = args.ip
    ))
    
    # Start the system
    base.start(
        pollEn   = False,
        initRead = False,
        zmqPort  = None,
    )

    xpm = base.XPM
    app = base.XPM.XpmApp

    # Print the AxiVersion Summary
    xpm.AxiVersion.printStatus()

    provider = StaticProvider(__name__)

    lock = Lock()
    pvstats = PVStats(provider, lock, args.P, xpm)
    pvctrls = PVCtrls(provider, lock, args.P, args.ip, xpm, pvstats._groups)

    # process PVA transactions
    updatePeriod = 1.0
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
