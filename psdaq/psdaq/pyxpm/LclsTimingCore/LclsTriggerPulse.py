#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue Timing trigger pulse configuration
#-----------------------------------------------------------------------------
# File       : LclsTriggerPulse.py
# Created    : 2017-04-12
#-----------------------------------------------------------------------------
# Description:
# PyRogue Timing trigger pulse configuration
#-----------------------------------------------------------------------------
# This file is part of the rogue software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the rogue software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import pyrogue as pr

class LclsTriggerPulse(pr.Device):
    def __init__(   self,       
            name        = "LclsTriggerPulse",
            description = "Timing trigger pulse configuration",
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        ##############################
        # Variables
        ##############################

        self.addRemoteVariables(   
            name         = "OpCodeMask",
            description  = "Opcode mask 256 bits to connect the pulse to any combination of opcodes",
            offset       =  0x00,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            number       =  8,
            stride       =  4,
        )

        self.add(pr.RemoteVariable(    
            name         = "PulseDelay",
            description  = "Pulse delay (Number of recovered clock cycles)",
            offset       =  0x20,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "PulseWidth",
            description  = "Pulse Width (Number of recovered clock cycles)",
            offset       =  0x24,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "PulsePolarity",
            description  = "Pulse polarity: 0-Normal. 1-Inverted",
            offset       =  0x28,
            bitSize      =  1,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))
        