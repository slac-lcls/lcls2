#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue LCLS-I EVR Registers
#-----------------------------------------------------------------------------
# File       : EvrV1Reg.py
# Created    : 2017-04-12
#-----------------------------------------------------------------------------
# Description:
# PyRogue LCLS-I EVR Registers
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

class EvrV1Reg(pr.Device):
    def __init__(   self,       
            name        = "EvrV1Reg",
            description = "LCLS-I EVR Registers",
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        ##############################
        # Variables
        ##############################

        self.add(pr.RemoteVariable(    
            name         = "Status",
            description  = "Status Register",
            offset       =  0x00,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "Control",
            description  = "Control Register",
            offset       =  0x04,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "IrqFlagWr",
            description  = "Interrupt Flag Register",
            offset       =  0x08,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "WO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "IrqFlagRd",
            description  = "Interrupt Flag Register",
            offset       =  0x08,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))                        

        self.add(pr.RemoteVariable(    
            name         = "IrqEnable",
            description  = "Interrupt Enable Register",
            offset       =  0x0C,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "PulseIrqMap",
            description  = "Mapping register for pulse interrupt",
            offset       =  0x10,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "PcieIntEna",
            description  = "PCIe interrupt Enable and state status",
            offset       =  0x14,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "FWVersion",
            description  = "Firmware Version Register",
            offset       =  0x2C,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "FWVersionUnmasked",
            description  = "Firmware Version without 0x1F mask and byte swapped",
            offset       =  0x30,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "UsecDivider",
            description  = "Divider to get from Event Clock to 1 MHz",
            offset       =  0x4C,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "SecSR",
            description  = "Seconds Shift Register",
            offset       =  0x5C,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "SecCounter",
            description  = "Timestamp Seconds Counter",
            offset       =  0x60,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "EventCounter",
            description  = "Timestamp Event Counter",
            offset       =  0x64,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "SecLatch",
            description  = "Timestamp Seconds Counter Latch",
            offset       =  0x68,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "EvCntLatch",
            description  = "Timestamp Event Counter Latch",
            offset       =  0x6C,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "IntEventEn",
            description  = "Internal Event Enable",
            offset       =  0xA0,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "IntEventCount",
            description  = "Internal Event Count",
            offset       =  0xA4,
            bitSize      =  1,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "IntEventCode",
            description  = "Internal Event Code",
            offset       =  0xA8,
            bitSize      =  8,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "ExtEventEn",
            description  = "External Event Enable",
            offset       =  0xAC,
            bitSize      =  1,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "ExtEventCode",
            description  = "External Event Code",
            offset       =  0xB0,
            bitSize      =  8,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.addRemoteVariables(   
            name         = "Pulse00",
            description  = "Pulse 0 Registers",
            offset       =  0x200,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            number       =  4,
            stride       =  4,
        )

        self.addRemoteVariables(   
            name         = "Pulse01",
            description  = "Pulse 1 Registers",
            offset       =  0x210,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            number       =  4,
            stride       =  4,
        )

        self.addRemoteVariables(   
            name         = "Pulse02",
            description  = "Pulse 2 Registers",
            offset       =  0x220,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            number       =  4,
            stride       =  4,
        )

        self.addRemoteVariables( 
            name         = "Pulse03",
            description  = "Pulse 3 Registers",
            offset       =  0x230,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            number       =  4,
            stride       =  4,
        )

        self.addRemoteVariables(  
            name         = "Pulse04",
            description  = "Pulse 4 Registers",
            offset       =  0x240,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            number       =  4,
            stride       =  4,
        )

        self.addRemoteVariables(  
            name         = "Pulse05",
            description  = "Pulse 5 Registers",
            offset       =  0x250,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            number       =  4,
            stride       =  4,
        )

        self.addRemoteVariables(  
            name         = "Pulse06",
            description  = "Pulse 6 Registers",
            offset       =  0x260,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            number       =  4,
            stride       =  4,
        )

        self.addRemoteVariables( 
            name         = "Pulse07",
            description  = "Pulse 7 Registers",
            offset       =  0x270,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            number       =  4,
            stride       =  4,
        )

        self.addRemoteVariables(   
            name         = "Pulse08",
            description  = "Pulse 8 Registers",
            offset       =  0x280,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            number       =  4,
            stride       =  4,
        )

        self.addRemoteVariables(   
            name         = "Pulse09",
            description  = "Pulse 9 Registers",
            offset       =  0x290,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            number       =  4,
            stride       =  4,
        )

        self.addRemoteVariables( 
            name         = "Pulse10",
            description  = "Pulse 10 Registers",
            offset       =  0x2A0,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            number       =  4,
            stride       =  4,
        )

        self.addRemoteVariables(  
            name         = "Pulse11",
            description  = "Pulse 11 Registers",
            offset       =  0x2B0,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            number       =  4,
            stride       =  4,
        )

        self.addRemoteVariables(   
            name         = "OutputMap",
            description  = "Front Panel Output Map Registers [11:0]",
            offset       =  0x440,
            bitSize      =  16,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            number       =  12,
            stride       =  4,
        )

        # self.addRemoteVariables(   
            # name         = "MapRam1",
            # description  = "Event Mapping RAM 1 [1023:0]",
            # offset       =  0x4000,
            # bitSize      =  32,
            # bitOffset    =  0x00,
            # base         = pr.UInt,
            # mode         = "RO",
            # number       =  1024,
            # stride       =  4,
            # hidden       =  True,
        # )
                     
        # self.addRemoteVariables(   
            # name         = "MapRam2",
            # description  = "Event Mapping RAM 2 [1023:0]",
            # offset       =  0x6000,
            # bitSize      =  32,
            # bitOffset    =  0x00,
            # base         = pr.UInt,
            # mode         = "RO",
            # number       =  1024,
            # stride       =  4,
            # hidden       =  True,
        # )
                        