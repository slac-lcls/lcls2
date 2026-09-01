#!/usr/bin/env python3
##############################################################################
## This file is part of 'PGP PCIe APP DEV'.
## It is subject to the license terms in the LICENSE.txt file found in the
## top-level directory of this distribution and at:
##    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
## No part of 'PGP PCIe APP DEV', including this file,
## may be copied, modified, propagated, or distributed except according to
## the terms contained in the LICENSE.txt file.
##############################################################################

import sys
import rogue
import numpy
import logging
import argparse
import time

import rogue.hardware.axi
import rogue.interfaces.stream

import pyrogue as pr
import pyrogue.pydm
import pyrogue.utilities.prbs

import axipcie            as pcie
import surf.axi           as axi
import surf.protocols.pgp as pgp

from p4p.server.thread import SharedPV
from p4p.nt import NTScalar
import psdaq.cas.pvhandler

from p4p.server import Server, StaticProvider

provider = None

class MyProvider(StaticProvider):
    def __init__(self, name):
        super(MyProvider,self).__init__(name)
        self.pvdict = {}

    def add(self,name,pv):
        self.pvdict[name] = pv
        super(MyProvider,self).add(name,pv)

def detect_version(dev):
    ''' Detect if the board is a C1100 by reading /proc/datadev_0 '''
    file_datadev='/proc/'+dev.split('/')[-1]
    boardType = None
    version = None

    with open(file_datadev, 'r', encoding='utf-8') as file:
        for line in file:
            if 'Build String' in line:
                index0 = line.find(':')
                index1 = line.find('Pgp')
                boardType = line[index0+2:index1]
                if index1 >= 0:
                    version = int(line[index1+3])
                break
        return boardType, version

class PgpMonitor(pr.Root):
    def __init__(   self,
                    name        = "pciServer",
                    description = "DMA Loopback Testing",
                    dev         = '/dev/datadev_0',
                    lanemask    = 0xf,
                    numVc       = 4,
                    pv          = None,
                    **kwargs):
        super().__init__(timeout=5.0, **kwargs)
        self.lanemask = lanemask
        self.numlane  = int(numpy.log2(lanemask))+1
        self.pv       = pv
        self.stat = {a:{} for a in range(self.numlane)}

        boardType, pgpversion = detect_version(dev)

        self.zmqServer = pyrogue.interfaces.ZmqServer(root=self, addr='127.0.0.1', port=0)
        self.addInterface(self.zmqServer)

        # Create PCIE memory mapped interface
        self.memMap = rogue.hardware.axi.AxiMemMap(dev)

        # Add the PCIe core device to base
        self.add(pcie.AxiPcieCore(
            offset     = 0x00000000,
            memBase     = self.memMap,
            numDmaLanes = self.numlane,
            boardType   = boardType,
            expand      = True,
        ))

        # Add PGP Core
        for lane in range(self.numlane):
            if (pgpversion == 4):
                self.add(pgp.Pgp4AxiL(
                    name    = f'Lane[{lane}]',
                    offset  = (0x00800000 + lane*0x00010000),
                    memBase = self.memMap,
                    numVc   = numVc,
                    writeEn = True,
                    expand  = True,
                ))
            elif (pgpversion == 3):
                self.add(pgp.Pgp3AxiL(
                    name    = f'Lane[{lane}]',
                    offset  = (0x00800000 + lane*0x00010000),
                    memBase = self.memMap,
                    numVc   = numVc,
                    writeEn = True,
                    expand  = False,
                ))
            else:
                self.add(pgp.Pgp2bAxi(
                    name    = f'Lane[{lane}]',
                    offset  = (0x00800000 + lane*0x00010000 + 0x1000),
                    memBase = self.memMap,
                    expand  = False,
                ))

            self.add(axi.AxiStreamMonAxiL(
                name        = (f'PgpTxAxisMon[{lane}]'),
                offset      = (0x00800000 + lane*0x00010000 + 0x3000),
                numberLanes = numVc,
                memBase     = self.memMap,
                expand      = False,
            ))

            self.add(axi.AxiStreamMonAxiL(
                name        = (f'PgpRxAxisMon[{lane}]'),
                offset      = (0x00800000 + lane*0x00010000 + 0x4000),
                numberLanes = numVc,
                memBase     = self.memMap,
                expand      = False,
            ))
        for lane in range(self.numlane):
            self.add(axi.AxiStreamDmaV2Fifo(
                name        = (f'AxiStreamDmaV2Fifo[{lane}]'),
                offset      = (0x0010_0000 + lane*0x100),
                memBase     = self.memMap,
                expand      = False,
            ))

    def init_lanes(self):

        linkReadyMask = 0
        remRxLinkReadyMask = 0

        for i in range(self.numlane):
            if (1<<i)&self.lanemask:
                lane = getattr(self,f'Lane[{i}]')
                if lane is None:
                    logging.critical(f'Error looking up Lane[{i}]')
                def checkRegBool(name, mask):
                    attr = getattr(lane.RxStatus,name)
                    val = attr.get()
                    if val==False:
                        logging.critical(f'Lane{i}.RxStatus.{name}: {val}')
                        raise RuntimeError(f'Lane{i}.RxStatus.{name}: {val}')
                    else:
                        mask += 1<<i
                checkRegBool('LinkReady',linkReadyMask)
                checkRegBool('RemRxLinkReady',remRxLinkReadyMask)

                def initReg(name):
                    val = getattr(lane.RxStatus,name).get()
                    self.stat[i][name] = val
                    if self.pv is not None:
                        setattr(self, f'_pv_lane{i}_{name}', addPv(f'{self.pv}:Lane{i}:{name}','I', val))

                for r in ('LinkErrorCnt','LinkDownCnt','LinkReadyCnt','RemRxLinkReadyCnt'):
                    initReg(r)

        if self.pv is not None:
            self._pv_linkReadyMask      = addPv(f'{self.pv}:LinkReady','I',linkReadyMask)
            self._pv_remRxLinkReadyMask = addPv(f'{self.pv}:RemRxLinkReady','I',remRxLinkReadyMask)

    def check_lanes(self,header=''):

        timev = divmod(float(time.time_ns()), 1.0e9)

        linkReadyMask = 0
        remRxLinkReadyMask = 0

        for i in range(self.numlane):
            if (1<<i)&self.lanemask:
                lane = getattr(self,f'Lane[{i}]')

                def checkRegBool(name, mask):
                    val = getattr(lane.RxStatus,name).get()
                    if val==False:
                        logging.critical(f'{header}: Lane{i}.RxStatus.{name}: {val}')
                        raise RuntimeError(f'{header}: Lane{i}.RxStatus.{name}: {val}')
                    else:
                        mask += 1<<i
                checkRegBool('LinkReady'     ,linkReadyMask)
                checkRegBool('RemRxLinkReady',remRxLinkReadyMask)

                def checkReg(name):
                    val = getattr(lane.RxStatus,name).get()
                    if self.pv is not None:
                        updatePv(getattr(self,f'_pv_lane{i}_{name}'), val, timev)
                    if val != self.stat[i][name]:
                        logging.warning(f'{header}: Lane{i}.RxStatus.{name}: {stat[i][name]} -> {val}')
                        self.stat[i][name] = val

                for r in ('LinkErrorCnt','LinkDownCnt','LinkReadyCnt','RemRxLinkReadyCnt'):
                    checkReg(r)

        if self.pv is not None:
            updatePv(self._pv_linkReadyMask     , linkReadyMask     , timev)
            updatePv(self._pv_remRxLinkReadyMask, remRxLinkReadyMask, timev)

def main():
    global pvdb
    global provider

    parser = argparse.ArgumentParser(description='Monitor PGP links and export to EPICS')
    parser.add_argument('--dev'  , help='device file name (/dev/datadev_0)', type=str, default='/dev/datadev_0')
    parser.add_argument('--lanes', help='bit mask of lanes (0xf)', type=int, default=0xf)
    parser.add_argument('--pv',    help='base name for PVs (LOCAL:PGPMON)', type=str, default=None)
    args = parser.parse_args()

    pvdb     = {}
    provider = MyProvider(__name__)

    pgpmon   = PgpMonitor( dev      = args.dev,
                           lanemask = args.lanes,
                           numVc    = 1,
                           pv       = args.pv )
    pgpmon.__enter__()

    with Server(providers=[provider]):
        pgpmon.check_lanes()
        time.sleep(10)

if __name__ == "__main__":
    main()
