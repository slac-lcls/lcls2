#!/usr/bin/env python3
##############################################################################
## This file is part of 'EPIX'.
## It is subject to the license terms in the LICENSE.txt file found in the 
## top-level directory of this distribution and at: 
##    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
## No part of 'EPIX', including this file, 
## may be copied, modified, propagated, or distributed except according to 
## the terms contained in the LICENSE.txt file.
##############################################################################

import pyrogue as pr
import pyrogue.protocols
import time
import xpm
import kcu
import LclsTimingCore as timing
from _AxiLiteRingBuffer import AxiLiteRingBuffer

class Top(pr.Device):

    def __init__(   self,       
            name        = "Top",
            description = "Container for XPM",
            memBase     = 0,
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)
        
        ######################################################################
        
        # Add devices
        self.add(kcu.AxiPcieCore(
            boardType = 'Kcu1500',
            memBase = memBase,
            offset  = 0x00000000, 
            expand  = False,
        ))

        self.add(xpm.XpmApp(
            memBase = memBase,
            name   = 'XpmApp',
            offset = 0x00800000,
        ))
        
        self.add(AxiLiteRingBuffer(
            memBase = memBase,
            name      = 'AxiLiteRingBuffer',
            datawidth = 16,
            offset    = 0x00810000,
        ))

        self.add(xpm.XpmSequenceEngine(
            memBase = memBase,
            name   = 'SeqEng_0',
            offset = 0x00820000,
        ))

        self.add(timing.TpgMiniCore(
            memBase = memBase,
            name   = 'TpgMini',
            offset = 0x00830000,
        ))

#        self.add(xpm.CuPhase(
#            memBase = memBase,
#            name = 'CuPhase',
#            offset = 0x00850000,
#        ))

        self.add(xpm.XpmPhase(
            memBase = memBase,
            name   = 'CuToScPhase',
            offset = 0x00850000,
        ))

    def start(self):
        #  Reprogram the reference clock
        self.AxiPcieCore.I2cMux.set(1<<2)
        self.AxiPcieCore.Si570._program()
        return
        time.sleep(0.01)
        #  Reset the Tx and Rx PLLs
        for i in range(8):
            self.XpmApp.link.set(i)
            self.XpmApp.txPllReset.set(1)
            time.sleep(0.01)
            self.XpmApp.txPllReset.set(0)
            time.sleep(0.01)
            self.XpmApp.rxPllReset.set(1)
            time.sleep(0.01)
            self.XpmApp.rxPllReset.set(0)
        self.XpmApp.link.set(0)

