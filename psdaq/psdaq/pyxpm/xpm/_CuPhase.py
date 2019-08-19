#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue Phase Measurement Module
#-----------------------------------------------------------------------------
# File       : CuPhase.py
# Created    : 2019-06-24
#-----------------------------------------------------------------------------
# Description:
# PyRogue Phase Measurement Module
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

class CuPhase(pr.Device):
    def __init__(   self, 
            name        = "CuPhase", 
            description = "XPM Cu Phase Measurement Module", 
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        ##############################
        # Variables
        ##############################

        self.add(pr.RemoteVariable(    
            name         = "base",
            description  = "Base subharmonic",
            offset       =  0x00,
            bitSize      =  20,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(
            name         = "early",
            description  = "Early samples",
            offset       =  0x04,
            bitSize      =  20,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(
            name         = "gate",
            description  = "Base gate integral",
            offset       =  0x08,
            bitSize      =  20,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(
            name         = "late",
            description  = "Late samples",
            offset       =  0x0c,
            bitSize      =  20,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        def _phaseValue(var,read):
            v = 0
            s = var.dependencies[0].get(read)
            if s>0:
                t = var.dependencies[1].get(read)
                v = float(t)/float(s)*1000./119.   # ns
            return v

        self.add(pr.LinkVariable(    
            name         = "phase_ns",
            description  = "Phase value in ns",
            linkedGet    = _phaseValue,
            dependencies = [self.base,self.early]
        ))


        def _baseWidth(var,read):
            v = 0
            s = var.dependencies[0].get(read)
            if s>0:
                t = var.dependencies[1].get(read)
                v = float(t)/float(s)*1000./119.   # ns
            return v

        self.add(pr.LinkVariable(    
            name         = "base_ns",
            description  = "Base gate width in ns",
            linkedGet    = _baseWidth,
            dependencies = [self.base,self.gate]
        ))


