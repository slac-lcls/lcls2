#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue XPM Timing Delay Application Module
#-----------------------------------------------------------------------------
# File       : TimingDelayAppl.py
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
import psdaq.pyxpm.LclsTimingCore               as timing

class BsasModule(pr.Device):

    def __init__(   self, 
            name        = "BsasControl", 
            description = "HPS BLD Application Module", 
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        ##############################
        # Variables
        ##############################
        self.add(pr.RemoteVariable(    
            name         = 'Enable',
            description  = "Enable",
            offset       =  0x100,
            bitSize      =  1,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'channelMask',
            description  = "Channel Mask",
            offset       =  0x104,
            bitSize      =  31,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'channelSevr',
            description  = "Max severity by channel",
            offset       =  0x108,
            bitSize      =  64,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))
        self.add(timing.EvrV2ChannelReg(    
            name         = 'acquire',
            description  = "Acquire Trigger",
            offset       =  0x400,
        ))
        self.add(timing.EvrV2ChannelReg(    
            name         = 'rowAdvance',
            description  = "Row Advance Trigger",
            offset       =  0x500,
        ))
        self.add(timing.EvrV2ChannelReg(    
            name         = 'tableReset',
            description  = "Table Reset Trigger",
            offset       =  0x600,
        ))

class BsasControl(pr.Device):

    def __init__(   self, 
                    name        = "BsasControl", 
                    description = "HPS BLD Application Module", 
                    numEdefs    = 4,
                    **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        for i in range(numEdefs):
            self.add(BsasModule(
                name = f'Bsas{i}',
                offset = 0x800*i,
            ))
