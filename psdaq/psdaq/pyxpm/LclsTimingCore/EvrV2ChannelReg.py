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
            dmaEnable   = False,
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)
        #########################################################  
        self.add(pr.RemoteVariable(
            name        = "EnableReg",
            description = "Enable Register",
            offset      = 0x00,
            bitSize     = 1,
            bitOffset   = 0,
            base        = pr.Bool,
            mode        = "RW",
        ))
        #########################################################
        if (dmaEnable):
            self.add(pr.RemoteVariable(
                name        = "BsaEnabled",
                description = "BSA Enable register (Only valid register is DMA_ENABLE_G=true)",
                offset      = 0x00,
                bitSize     = 1,
                bitOffset   = 1,
                base        = pr.Bool,
                mode        = "RW",
            ))
        #########################################################  
            self.add(pr.RemoteVariable(
                name        = "dmaEnabled",
                description = "DMA Enable register (Only valid register is DMA_ENABLE_G=true)",
                offset      = 0x00,
                bitSize     = 1,
                bitOffset   = 2,
                base        = pr.Bool,
                mode        = "RW",
            ))
        ########################################################  
        self.add(pr.RemoteVariable(
            name        = "RateSel",
            description = "Rate select",
            offset      = 0x04,
            bitSize     = 13,
            bitOffset   = 0,
            mode        = "RW",
        ))
        #########################################################  
        self.add(pr.RemoteVariable(
            name        = "DestSel",
            description = "Destination select",
            offset      = 0x04,
            bitSize     = 19,
            bitOffset   = 13,
            mode        = "RW",
        ))
        #########################################################  
        self.add(pr.RemoteVariable(
            name        = "Count",
            description = "Counts",
            offset      = 0x08,
            bitSize     = 32,
            bitOffset   = 0,
            mode        = "RO",
            pollInterval = 1,
        ))
        #########################################################  
        if (dmaEnable):
            self.add(pr.RemoteVariable(
                name        = "BsaWindowDelay",
                description = "Start of BSA sensitivity window following trigger (Only valid register is DMA_ENABLE_G=true)",
                offset      = 0x0C,
                bitSize     = 20,
                bitOffset   = 0,
                mode        = "RW",
            ))
        ######################################################### 
            self.add(pr.RemoteVariable(
                name        = "BsaWindowSetup",
                description = "Start of BSA sensitivity window before trigger (Only valid register is DMA_ENABLE_G=true)",
                offset      = 0x0C,
                bitSize     = 6,
                bitOffset   = 20,
                mode        = "RW",
            ))
        #########################################################
            self.add(pr.RemoteVariable(
                name        = "BsaWindowWidth",
                description = "Width of BSA sensitivity window (Only valid register is DMA_ENABLE_G=true)",
                offset      = 0x10,
                bitSize     = 20,
                bitOffset   = 0,
                mode        = "RW",
            ))
        #########################################################