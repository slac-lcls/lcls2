#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue XPM Phase Measurement
#-----------------------------------------------------------------------------
# File       : XpmPhase.py
# Created    : 2019-10-02
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

class XpmPhase(pr.Device):
    def __init__(   self, 
            name        = "XpmPhase", 
            description = "Phase Measurement", 
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        self.add(pr.RemoteVariable(    
            name         = "block",
            description  = "Status registers",
            offset       =  0x00,
            bitSize      =  32*4,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

    def phase(self):
        v = self.block.get()
        m = (1<<32)-1
        n = (v>> 0)&m
        e = (v>>32)&m
        l = (v>>96)&m
        return e/(e+l)*8/119e6

