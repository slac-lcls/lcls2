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
            name         = "delayValue",
            description  = "Delay value from scan",
            offset       =  0x04,
            bitSize      =  11,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "delayEnd",
            description  = "Delay value scan range",
            offset       =  0x06,
            bitSize      =  11,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "External Lock",
            description  = "External lock status",
            offset       =  0x07,
            bitSize      =  1,
            bitOffset    =  0x05,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "reset",
            description  = "Reset status",
            offset       =  0x07,
            bitSize      =  1,
            bitOffset    =  0x06,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "Internal Lock",
            description  = "Internal lock status",
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
        ))

        @self.command(name="Rescan", description="Reset and rescan",)
        def Rescan():
            self.rescan.set(1)

