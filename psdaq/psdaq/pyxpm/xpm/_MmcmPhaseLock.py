#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue MMCM Phase Lock Module
#-----------------------------------------------------------------------------
# File       : MmcmPhaseLock.py
# Created    : 2019-06-24
#-----------------------------------------------------------------------------
# Description:
# PyRogue MMCM Phase Lock Module
#-----------------------------------------------------------------------------
# This file is part of the rogue software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the rogue software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import pyrogue        as pr

class MmcmPhaseLock(pr.Device):
    def __init__(   self, 
            name        = "MmcmPhaseLock", 
            description = "XPM MMCM Phase Lock Module", 
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        ##############################
        # Variables
        ##############################

        self.add(pr.RemoteVariable(    
            name         = "delaySet",
            description  = "Delay setting",
            offset       =  0x00,
            bitSize      =  11,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "bypassLock",
            description  = "Bypass external lock",
            offset       =  0x00,
            bitSize      =  2,
            bitOffset    =  0x10,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(
            name         = "delayValue",
            description  = "delayValue",
            offset       =  0x04,
            bitSize      =  11,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(
            name         = "delayEnd",
            description  = "delayEnd",
            offset       =  0x08,
            bitSize      =  11,
            bitOffset    =  0x10,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(
            name         = "externalLock",
            description  = "externalLock",
            offset       =  0x07,
            bitSize      =  1,
            bitOffset    =  0x05,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(
            name         = "nready",
            description  = "nready",
            offset       =  0x07,
            bitSize      =  1,
            bitOffset    =  0x06,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(
            name         = "internalLock",
            description  = "internalLock",
            offset       =  0x07,
            bitSize      =  1,
            bitOffset    =  0x07,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "ramAddr",
            description  = "Readback RAM address",
            offset       =  0x08,
            bitSize      =  11,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "ramData",
            description  = "Readback RAM value",
            offset       =  0x0C,
            bitSize      =  6,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "rescan",
            description  = "Restart scan",
            offset       =  0x10,
            bitSize      =  1,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            verify       = False,
        ))

        @self.command(name="Rescan", description="Reset and rescan",)
        def Rescan():
            self.rescan.set(1)

        self.add(pr.RemoteVariable(    
            name         = "halfPeriod",
            description  = "Half period",
            offset       =  0x14,
            bitSize      =  16,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "sumPeriod",
            description  = "Sum period",
            offset       =  0x14,
            bitSize      =  16,
            bitOffset    =  16,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "minSum",
            description  = "min Sum",
            offset       =  0x18,
            bitSize      =  16,
            bitOffset    =  0,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "minDelay",
            description  = "min Delay",
            offset       =  0x18,
            bitSize      =  16,
            bitOffset    =  16,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "maxSum",
            description  = "max Sum",
            offset       =  0x1c,
            bitSize      =  16,
            bitOffset    =  0,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "maxDelay",
            description  = "max Delay",
            offset       =  0x1c,
            bitSize      =  16,
            bitOffset    =  16,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "ramData1",
            description  = "Readback RAM1 value",
            offset       =  0x20,
            bitSize      =  16,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

    def waveform(self):
        w = [0]*2048
        for i in range(2048):
            self.ramAddr.set(i)
            w[i] = self.ramData.get()
        return w

    def dump(self):
        print(f'{self.name}:')
        print(f'bypassLock: {self.bypassLock.get()}')
#        print(f'status    : {self.status.get()}')
        print(f'nready    : {self.nready.get()}')
        print(f'externalL : {self.externalLock.get()}')
        print(f'internalL : {self.internalLock.get()}')
