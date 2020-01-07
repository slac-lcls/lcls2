#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue XPM Cu Generator Module
#-----------------------------------------------------------------------------
# File       : CuGenerator.py
# Created    : 2019-06-24
#-----------------------------------------------------------------------------
# Description:
# PyRogue XPM Cu Generator Module
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

class CuGenerator(pr.Device):
    def __init__(   self, 
            name        = "CuGenerator", 
            description = "XPM Cu Generator Module", 
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        ##############################
        # Variables
        ##############################

        self.add(pr.RemoteVariable(    
            name         = "timeStamp",
            description  = "Received time stamp",
            offset       =  0x00,
            bitSize      =  64,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "pulseId",
            description  = "Received pulse ID",
            offset       =  0x08,
            bitSize      =  64,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "cuDelay",
            description  = "Retransmission delay in 186MHz clks",
            offset       =  0x10,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "cuBeamCode",
            description  = "Eventcode for Beam present translation",
            offset       =  0x14,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "cuFiducialIntv",
            description  = "Interval between last two Cu fiducials",
            offset       =  0x18,
            bitSize      =  19,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            verify       = False,
        ))

        self.add(pr.RemoteVariable(    
            name         = "cuFiducialIntvErr",
            description  = "Latched error from Cu fiducial interval",
            offset       =  0x1B,
            bitSize      =  1,
            bitOffset    =  0x7,
            base         = pr.UInt,
            mode         = "RW",
            verify       = False,
        ))

        @self.command(name="ClearFiducialErr", description="Clear the fiducial error latch",)
        def ClearFiducialErr():
            self.cuFiducialIntv.set(0)

    def timeStampSec(self):
        ts = self.timeStamp.get()
        if ts is not None:
            ts >>= 32
        return ts
