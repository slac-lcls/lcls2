#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue LCLS-I EVR ISR Controller
#-----------------------------------------------------------------------------
# File       : EvrV1Isr.py
# Created    : 2017-04-12
#-----------------------------------------------------------------------------
# Description:
# PyRogue LCLS-I EVR ISR Controller
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

class EvrV1Isr(pr.Device):
    def __init__(   self,       
            name        = "EvrV1Isr",
            description = "LCLS-I EVR ISR Controller",
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        ##############################
        # Variables
        ##############################

        self.add(pr.RemoteVariable(    
            name         = "IsrSelect",
            description  = "0x1 = Software ISR, 0x0 = Firmware ISR",
            offset       =  0x00,
            bitSize      =  1,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

