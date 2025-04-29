#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue XPM DestDiagControl Module
#-----------------------------------------------------------------------------
# File       : XpmSequenceEngine.py
# Created    : 2019-06-24
#-----------------------------------------------------------------------------
# Description:
# PyRogue XPM  Module
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

class DestDiagControl(pr.Device):
    def __init__(   self, 
                    name        = "DestDiagControl", 
                    description = "XPM rate-limited destination codes", 
                    **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        self.add(pr.RemoteVariables(    
            name         = "interval",
            description  = "Minimum interval",
            offset       =  0x00,
            bitSize      =  20,
            bitOffset    =  0x00,
            stride       = 32,
            number       = 4,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariables(    
            name         = "count",
            description  = "Count",
            offset       =  0x04,
            bitSize      =  32,
            bitOffset    =  0x00,
            stride       = 32,
            number       = 4,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "DestMask",
            description  = "Destination mask",
            offset       =  0x20,
            bitSize      =  16,
            bitOffset    =  0x0,
            base         = pr.UInt,
            mode         = "RW",
        ))


