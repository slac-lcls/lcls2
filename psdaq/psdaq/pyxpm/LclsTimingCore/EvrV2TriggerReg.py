#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue LCLS-II EVR V2 Trigger Registers
#-----------------------------------------------------------------------------
# File       : Device.py
# Created    : 2018-09-17
#-----------------------------------------------------------------------------
# Description:
# PyRogue LCLS-II EVR V2 Trigger Registers
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

class EvrV2TriggerReg(pr.Device):
    def __init__(   self,
            name        = "EvrV2TriggerReg",
            description = "EVR V2 Trigger",
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)
    #########################################################  
        self.add(pr.RemoteVariable(
            name        = "Enable",
            description = "Trigger Enable",
            offset      = 0x03,
            bitSize     = 1,
            bitOffset   = 7,
            base        = pr.UInt,
            mode        = "RW",
        ))
    #########################################################
        self.add(pr.RemoteVariable(
            name        = "Source",
            description = "Source mask",
            offset      = 0x00,
            bitSize     = 4,
            bitOffset   = 4,
            base        = pr.UInt,
            mode        = "RW",
        ))
    #########################################################  
        self.add(pr.RemoteVariable(
            name        = "Polarity",
            description = "Polarity",
            offset      = 0x02,
            bitSize     = 1,
            bitOffset   = 0,
            base        = pr.UInt,
            mode        = "RW",
        ))
    #########################################################  
        self.add(pr.RemoteVariable(
            name        = "Delay",
            description = "Delay in ticks",
            offset      = 0x04,
            bitSize     = 28,
            bitOffset   = 0,
            base        = pr.UInt,
            mode        = "RW",
        ))
    #########################################################  
        self.add(pr.RemoteVariable(
            name        = "Width",
            description = "Width in ticks",
            offset      = 0x08,
            bitSize     = 28,
            bitOffset   = 0,
            base        = pr.UInt,
            mode        = "RW",
        ))
    #########################################################  
        self.add(pr.RemoteVariable(
            name        = "DelayTap",
            description = "Delay tpa in ticks/64",
            offset      = 0x0C,
            bitSize     = 6,
            bitOffset   = 0,
            base        = pr.UInt,
            mode        = "RW",
        ))
    #########################################################  
        self.add(pr.RemoteVariable(
            name        = "DelayTapReadback",
            description = "Delay tap readback in ticks/64",
            offset      = 0x0E,
            bitSize     = 6,
            bitOffset   = 0,
            base        = pr.UInt,
            mode        = "RW",
        ))
