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

class DevReset(pr.Device):
    def __init__(   self, 
            name        = "DevReset", 
            description = "Device Reset Control", 
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        self.add(pr.RemoteVariable(
            name         = 'clearTimingPhyReset',
            description  = "Clear timingPhyRst",
            offset       = 0x0100,
            bitSize      =  1,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

class NoTimingFrameRx(pr.Device):
    def __init__(self,
                 name        = 'NoTimingFrameRx',
                 description = "Dummy container",
                 memBase     = 0,
                 **kwargs):
        super().__init__(name=name, description=description, **kwargs)
        v = ['sofCount','eofCount','FidCount','CrcErrCount','RxClkCount','RxRstCount',
             'RxDecErrCount','RxDspErrCount','BypassResetCount','BypassDoneCount','TxClkCount',
             'RxLinkUp']
        for i in v:
            self.add(pr.LocalVariable(name = i, mode = 'RO', value=0))

#    @self.command(name="C_RxReset", description="Reset Rx Link",)
    def C_RxReset():
        pass

#    @self.command(name="ClearRxCounters", description="Clear the Rx status counters",)
    def ClearRxCounters():
        pass

    def update(self):
        pass

class NoCuGenerator(pr.Device):
    def __init__(self,
                 name        = 'NoCuGenerator',
                 description = "Dummy container",
                 memBase     = 0,
                 **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        v = ['pulseId','cuFiducialIntv','cuFiducialIntvErr','cuDelay','cuBeamCode']
        for i in v:
            self.add(pr.LocalVariable(name = i, mode = 'RO', value=0))

    def timeStampSec(self):
        return 0

class NoAxiSy56040(pr.Device):
    def __init__(self,
                 name        = 'NoAxiSy56040',
                 description = "Dummy container",
                 memBase     = 0,
                 **kwargs):
        super().__init__(name=name, description=description, **kwargs)

#        self.addNodes(pr.LocalVariable, number=4, stride=1, name = 'OutputConfig', value=0, offset=0)
        v = ['OutputConfig[0]','OutputConfig[1]','OutputConfig[2]','OutputConfig[3]']
        for i in v:
            self.add(pr.LocalVariable(name = i, mode = 'RO', value=0))

class NoMmcmPhaseLock(pr.Device):
    def __init__(   self, 
            name        = "NoMmcmPhaseLock", 
            description = "XPM MMCM Phase Lock Module", 
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        v = ['delaySet','bypassLock','status','delayValue',
             'externalLock','nready','internalLock','sumPeriod',
             'ramData','rescan','ramData1']
        for i in v:
            self.add(pr.LocalVariable(name = i, mode = 'RO', value=0))
        self.add(pr.LocalVariable(name = 'ramAddr', mode='RW', value=0))
        self.add(pr.LocalVariable(name = 'delayEnd', mode='RO', value=32))

    def dump(self):
        pass

class DevPcie(pr.Device):

    mmcmParms = [ ['MmcmPL119', 0x08900000],
                  ['MmcmPL70' , 0x08a00000],
                  ['MmcmPL130', 0x08b00000],
                  ['MmcmPL186', 0x80040000] ]

    def __init__(   self,       
            name        = "DevPcie",
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

        self.add(NoTimingFrameRx(
            name = 'UsTiming',
        ))

        self.add(NoTimingFrameRx(
            name = 'CuTiming',
        ))

        self.add(NoCuGenerator(
            name = 'CuGenerator',
        ))

        for i in range(len(DevPcie.mmcmParms)):
            self.add(NoMmcmPhaseLock(
                name   = DevPcie.mmcmParms[i][0],
            ))
        
        self.add(NoAxiSy56040(
            name = 'AxiSy56040',
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

        self.add(timing.TPGMiniCore(
            memBase = memBase,
            name   = 'TpgMini',
            offset = 0x00830000,
        ))

        self.add(DevReset(
            memBase = memBase,
            name   = 'DevReset',
            offset = 0x00840000,
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
        print('---DevPcie.start---')
        self.DevReset.clearTimingPhyReset.set(0)
        #  Reprogram the reference clock
        self.AxiPcieCore.I2cMux.set(1<<2)
        self.AxiPcieCore.Si570._program()
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
        self.DevReset.clearTimingPhyReset.set(1)

        print(f'FixedRateDiv0 f{self.TpgMini.FixedRateDiv[0].get()}')

        #  Set the fixed rate markers
        self.TpgMini.FixedRateDiv[0].set(910000)
        self.TpgMini.FixedRateDiv[1].set( 91000)
        self.TpgMini.FixedRateDiv[2].set(  9100)
        self.TpgMini.FixedRateDiv[3].set(   910)
        self.TpgMini.FixedRateDiv[4].set(    91)
        self.TpgMini.FixedRateDiv[5].set(    13)
        self.TpgMini.FixedRateDiv[6].set(     1)
        self.TpgMini.RateReload.set(1)

        print(f'FixedRateDiv0 f{self.TpgMini.FixedRateDiv[0].get()}')
