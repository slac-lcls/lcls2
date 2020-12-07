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
            name         = "status",
            description  = "Lock status",
            offset       =  0x04,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        def _delayValue(var,read):
            return (var.dependencies[0].get(read)>> 0)&0x7ff

        def _delayEnd(var,read):
            return (var.dependencies[0].get(read)>>16)&0x7ff

        def _externalLock(var,read):
            return (var.dependencies[0].get(read)>>29)&1

        def _nready(var,read):
            return (var.dependencies[0].get(read)>>30)&1

        def _internalLock(var,read):
            return (var.dependencies[0].get(read)>>31)&1

        self.add(pr.LinkVariable(    
            name         = "delayValue",
            description  = "Delay value from scan",
            linkedGet    = _delayValue,
            dependencies = [self.status]
        ))

        self.add(pr.LinkVariable(    
            name         = "delayEnd",
            description  = "Delay value scan range",
            linkedGet    = _delayEnd,
            dependencies = [self.status]
        ))

        self.add(pr.LinkVariable(    
            name         = "externalLock",
            description  = "External lock status",
            linkedGet    = _externalLock,
            dependencies = [self.status]
        ))

        self.add(pr.LinkVariable(    
            name         = "nready",
            description  = "Reset status",
            linkedGet    = _nready,
            dependencies = [self.status]
        ))

        self.add(pr.LinkVariable(    
            name         = "internalLock",
            description  = "Internal lock status",
            linkedGet    = _internalLock,
            dependencies = [self.status]
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
