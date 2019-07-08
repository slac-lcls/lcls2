#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue LCLS-II Timing Receiver module
#-----------------------------------------------------------------------------
# File       : Device.py
# Created    : 2017-04-12
#-----------------------------------------------------------------------------
# Description:
# PyRogue LCLS-II Timing Receiver module
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

class EvrV2Core(pr.Device):
    def __init__(   self,       
            name        = "EvrV2Core",
            description = "LCLS-II Timing Receiver module",
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        ##############################
        # Variables
        ##############################

        self.add(pr.RemoteVariable(    
            name         = "IrqEnable",
            description  = "Interrupt Enable",
            offset       =  0x00,
            bitSize      =  1,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "IrqStatus",
            description  = "Interrupt Pending",
            offset       =  0x04,
            bitSize      =  1,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "LinkAddr",
            description  = "Physical link address",
            offset       =  0x08,
            bitSize      =  16,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "gtxDebug",
            description  = "Debug bits from link",
            offset       =  0x0C,
            bitSize      =  8,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "CountReset",
            description  = "Counter reset",
            offset       =  0x10,
            bitSize      =  1,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "ModeSel",
            description  = "Select LCLS-I/LCLS-II Trigger outputs (0/1)",
            offset       =  0x14,
            bitSize      =  1,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "DmaFullThr",
            description  = "Set threshold in bytes for asserting readout full",
            offset       =  0x18,
            bitSize      =  24,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "ChannelEnable",
            description  = "Enable readout channel",
            offset       =  0x20,
            bitSize      =  1,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.addRemoteVariables(    
            name         = "ChannelBsaEnable",
            description  = "Enable BSA channel",
            offset       =  0x20,
            bitSize      =  1,
            bitOffset    =  0x01,
            base         = pr.UInt,
            mode         = "RW",
            number       =  12,
            stride       =  32,
        )

        self.addRemoteVariables( 
            name         = "ChannelDmaEnable",
            description  = "",
            offset       =  0x20,
            bitSize      =  1,
            bitOffset    =  0x02,
            base         = pr.UInt,
            mode         = "RW",
            number       =  12,
            stride       =  32,
        )

        self.add(pr.RemoteVariable(    
            name         = "ChannelRateSel",
            description  = "",
            offset       =  0x24,
            bitSize      =  13,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.addRemoteVariables(   
            name         = "ChannelDestSel",
            description  = "Channel event destination selection",
            offset       =  0x25,
            bitSize      =  18,
            bitOffset    =  0x05,
            base         = pr.UInt,
            mode         = "RW",
            number       =  12,
            stride       =  32,
        )

        self.addRemoteVariables(   
            name         = "ChannelEventCnt",
            description  = "Channel event counts",
            offset       =  0x28,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            number       =  12,
            stride       =  32,
        )

        self.addRemoteVariables(    
            name         = "ChannelBsaDelay",
            description  = "Channel BSA active delay",
            offset       =  0x2C,
            bitSize      =  20,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            number       =  12,
            stride       =  32,
        )

        self.addRemoteVariables(    
            name         = "ChannelBsaSetup",
            description  = "Channel BSA active setup",
            offset       =  0x2E,
            bitSize      =  12,
            bitOffset    =  0x04,
            base         = pr.UInt,
            mode         = "RW",
            number       =  12,
            stride       =  32,
        )

        self.addRemoteVariables(  
            name         = "ChannelBsaWidth",
            description  = "Channel BSA active width",
            offset       =  0x30,
            bitSize      =  20,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            number       =  12,
            stride       =  32,
        )

        self.add(pr.RemoteVariable(    
            name         = "GlobalEventCnt",
            description  = "Global Event count",
            offset       =  0x1A8,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.addRemoteVariables(   
            name         = "TriggerChannel",
            description  = "Channel used for event selection",
            offset       =  0x200,
            bitSize      =  4,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            number       =  12,
            stride       =  16,
        )

        self.addRemoteVariables(   
            name         = "TriggerPolarity",
            description  = "Trigger polarity (0=negative, 1=positive)",
            offset       =  0x202,
            bitSize      =  1,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            number       =  12,
            stride       =  16,
        )

        self.addRemoteVariables(  
            name         = "TriggerEnable",
            description  = "Trigger enable",
            offset       =  0x203,
            bitSize      =  1,
            bitOffset    =  0x07,
            base         = pr.UInt,
            mode         = "RW",
            number       =  12,
            stride       =  16,
        )

        self.addRemoteVariables(  
            name         = "TriggerDelay",
            description  = "Trigger width (186MHz clocks)",
            offset       =  0x208,
            bitSize      =  28,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            number       =  12,
            stride       =  16,
        )

        self.addRemoteVariables(   
            name         = "TriggerFineDelay",
            description  = "Trigger fine delay (82.2 ps steps)",
            offset       =  0x20C,
            bitSize      =  6,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            number       =  12,
            stride       =  16,
        )
        