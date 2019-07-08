#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue LCLS-II EVR V2 Channel Registers
#-----------------------------------------------------------------------------
# File       : Device.py
# Created    : 2018-09-17
#-----------------------------------------------------------------------------
# Description:
# PyRogue LCLS-II EVR V2 Channel Registers
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

class EvrV2ChannelReg(pr.Device):
    def __init__(   self,
            name        = "EvrV2ChannelReg",
            description = "EVR V2 Channel",
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)
    #########################################################  
        self.add(pr.RemoteVariable(
            name        = "Enable",
            description = "Enable",
            offset      = 0x00,
            bitSize     = 1,
            bitOffset   = 0,
            base        = pr.UInt,
            mode        = "RW",
        ))
    #########################################################  
        self.add(pr.RemoteVariable(
            name        = "BsaEnabled",
            description = "BSA enable register",
            offset      = 0x00,
            bitSize     = 1,
            bitOffset   = 1,
            base        = pr.UInt,
            mode        = "RW",
        ))
    #########################################################  
        self.add(pr.RemoteVariable(
            name        = "DmaEnabled",
            description = "DMA enable register",
            offset      = 0x00,
            bitSize     = 1,
            bitOffset   = 2,
            base        = pr.UInt,
            mode        = "RW",
        ))
    #########################################################  
        self.add(pr.RemoteVariable(
            name        = "RateSel",
            description = "Rate selection",
            offset      = 0x04,
            bitSize     = 13,
            bitOffset   = 0,
            base        = pr.UInt,
            mode        = "RW",
        ))
    #########################################################  
        self.add(pr.RemoteVariable(
            name        = "DestSel",
            description = "Destination selection",
            offset      = 0x04,
            bitSize     = 19,
            bitOffset   = 13,
            base        = pr.UInt,
            mode        = "RW",
        ))
    #########################################################  
        self.add(pr.RemoteVariable(
            name        = "Count",
            description = "Counts",
            offset      = 0x08,
            bitSize     = 32,
            bitOffset   = 0,
            base        = pr.UInt,
            mode        = "RW",
        ))
    #########################################################  
        self.add(pr.RemoteVariable(
            name        = "BsaWindowDelay",
            description = "Start of BSA sensitivity window following trigger",
            offset      = 0x0C,
            bitSize     = 20,
            bitOffset   = 0,
            base        = pr.UInt,
            mode        = "RW",
        ))
    #########################################################  
        self.add(pr.RemoteVariable(
            name        = "BsaWindowSetup",
            description = "Start of BSA sensitivity window before trigger",
            offset      = 0x0E,
            bitSize     = 6,
            bitOffset   = 4,
            base        = pr.UInt,
            mode        = "RW",
        ))
    #########################################################  
        self.add(pr.RemoteVariable(
            name        = "BsaWindowWidth",
            description = "Width of BSA sensitivity window",
            offset      = 0x10,
            bitSize     = 20,
            bitOffset   = 0,
            base        = pr.UInt,
            mode        = "RW",
        ))

