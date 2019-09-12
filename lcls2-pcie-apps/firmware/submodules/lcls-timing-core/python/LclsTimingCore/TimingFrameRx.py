#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue Status of timing frame reception
#-----------------------------------------------------------------------------
# File       : TimingFrameRx.py
# Created    : 2017-04-12
#-----------------------------------------------------------------------------
# Description:
# PyRogue Status of timing frame reception
# Associated firmware: lcls-timing-core/LCLS-II/core/rtl/TimingRx.vhd
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
import time 

class TimingFrameRx(pr.Device):
    def __init__(   self,       
            name        = "TimingFrameRx",
            description = "Status of timing frame reception",
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        ##############################
        # Variables
        ##############################

        self.add(pr.RemoteVariable(    
            name         = "sofCount",
            description  = "Start of frame count",
            offset       =  0x00,
            bitSize      =  32,
            bitOffset    =  0x00,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.add(pr.RemoteVariable(    
            name         = "eofCount",
            description  = "End of frame count",
            offset       =  0x04,
            bitSize      =  32,
            bitOffset    =  0x00,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.add(pr.RemoteVariable(    
            name         = "FidCount",
            description  = "Valid frame count",
            offset       =  0x08,
            bitSize      =  32,
            bitOffset    =  0x00,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.add(pr.RemoteVariable(    
            name         = "CrcErrCount",
            description  = "CRC error count",
            offset       =  0x0C,
            bitSize      =  32,
            bitOffset    =  0x00,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.add(pr.RemoteVariable(    
            name         = "RxClkCount",
            description  = "Recovered clock count div 16",
            offset       =  0x10,
            bitSize      =  32,
            bitOffset    =  0x00,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.add(pr.RemoteVariable(    
            name         = "RxRstCount",
            description  = "Receive link reset count",
            offset       =  0x14,
            bitSize      =  32,
            bitOffset    =  0x00,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.add(pr.RemoteVariable(    
            name         = "RxDecErrCount",
            description  = "Receive 8b/10b decode error count",
            offset       =  0x18,
            bitSize      =  32,
            bitOffset    =  0x00,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.add(pr.RemoteVariable(    
            name         = "RxDspErrCount",
            description  = "Receive disparity error count",
            offset       =  0x1C,
            bitSize      =  32,
            bitOffset    =  0x00,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.add(pr.RemoteVariable(    
            name         = "RxCountReset",
            description  = "Reset receive counters",
            offset       =  0x20,
            bitSize      =  1,
            bitOffset    =  0x00,
            mode         = "WO",
            hidden       = True,
        ))

        self.add(pr.RemoteVariable(    
            name         = "RxLinkUp",
            description  = "Receive link status",
            offset       =  0x20,
            bitSize      =  1,
            bitOffset    =  0x01,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.add(pr.RemoteVariable(    
            name         = "RxPolarity",
            description  = "Invert receive link polarity",
            offset       =  0x20,
            bitSize      =  1,
            bitOffset    =  0x02,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "RxReset",
            description  = "Reset receive link",
            offset       =  0x20,
            bitSize      =  1,
            bitOffset    =  0x03,
            mode         = "WO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "ClkSel",
            description  = "Select LCLS-I/LCLS-II Timing",
            offset       =  0x20,
            bitSize      =  1,
            bitOffset    =  0x04,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "RxDown",
            description  = "Rx down latch status",
            offset       =  0x20,
            bitSize      =  1,
            bitOffset    =  0x05,
            mode         = "RW",
            verify       = False,
        ))

        self.add(pr.RemoteVariable(    
            name         = "BypassRst",
            description  = "Buffer bypass reset status",
            offset       =  0x20,
            bitSize      =  1,
            bitOffset    =  0x06,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(
            name         = "RxPllReset",
            description  = "Reset RX PLL",
            offset       = 0x20,
            bitSize      = 1,
            bitOffset    = 0x07,
            mode         = "WO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "MsgDelay",
            description  = "LCLS-II timing frame pipeline delay (186MHz clks)",
            offset       =  0x24,
            bitSize      =  20,
            bitOffset    =  0x00,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "TxClkCount",
            description  = "Transmit clock counter div 16",
            offset       =  0x28,
            bitSize      =  32,
            bitOffset    =  0x00,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.add(pr.RemoteVariable(    
            name         = "BypassDoneCount",
            description  = "Buffer bypass done count",
            offset       =  0x2C,
            bitSize      =  16,
            bitOffset    =  0x00,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.add(pr.RemoteVariable(    
            name         = "BypassResetCount",
            description  = "Buffer bypass reset count",
            offset       =  0x2C,
            bitSize      =  16,
            bitOffset    =  16,
            mode         = "RO",
            pollInterval = 1,
        ))

        ##############################
        # Commands
        ##############################
        @self.command(name="C_RxReset", description="Reset Rx Link",)
        def C_RxReset():
            self.RxReset.set(1)
            time.sleep(0.001)
            self.RxReset.set(0)    

        @self.command(name="ClearRxCounters", description="Clear the Rx status counters",)
        def ClearRxCounters():
            self.RxCountReset.set(1)
            time.sleep(0.001)
            self.RxCountReset.set(0)                         
            
    def hardReset(self):
        self.ClearRxCounters()
        self.RxDown.set(0)  

    def softReset(self):
        self.ClearRxCounters()
        self.RxDown.set(0)  

    def countReset(self):
        self.ClearRxCounters()
        self.RxDown.set(0)
        