#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue XPM Path Timer Module
#-----------------------------------------------------------------------------
# File       : XpmPathTimer.py
# Created    : 2026-01-08
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

class XpmPathTimer(pr.Device):
    def __init__(   self, 
                    name        = "XpmPathTimer",
                    description = "XPM Deadtime Path Timer",
                    **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        self.add(pr.RemoteVariable(    
            name         = "latched",
            description  = "Channels that latched",
            offset       =  0x00,
            bitSize      =  16,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.addRemoteVariables(    
            name         = "chan",
            description  = "Channel Path Time",
            offset       =  0x04,
            bitSize      =  16,
            bitOffset    =  0x00,
            stride       =  4,
            base         = pr.UInt,
            mode         = "RO",
            number       = 14 ,
        )

