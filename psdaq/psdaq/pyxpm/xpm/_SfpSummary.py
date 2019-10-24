#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue MPS SFP Summary
#-----------------------------------------------------------------------------
# File       : SfpSummary.py
# Created    : 2019-06-24
#-----------------------------------------------------------------------------
# Description:
# PyRogue Timing Delay Application Module
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

class SfpSummary(pr.Device):

    def __init__(   self, 
            name        = "TimingDelayAppl", 
            description = "Timing Delay Application Module", 
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        ##############################
        # Variables
        ##############################

        self.add(pr.RemoteVariable(    
            name         = 'rsel0',
            description  = "Rate select bit0",
            offset       =  0x00,
            bitSize      =  8,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = 'rsel1',
            description  = "Rate select bit1",
            offset       =  0x04,
            bitSize      =  8,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = 'txfault',
            description  = "Transmit Fault",
            offset       =  0x08,
            bitSize      =  8,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = 'los',
            description  = "Loss of Signal",
            offset       =  0x0c,
            bitSize      =  8,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = 'modabs',
            description  = "Module abs",
            offset       =  0x10,
            bitSize      =  8,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))
        
