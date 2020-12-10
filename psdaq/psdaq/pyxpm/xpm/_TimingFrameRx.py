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

class TimingBitField(object):

    _block = 0

    def __init__(  self, name, description, offset, bitSize, bitOffset, **kwargs):
        self._name      = name
        self._offset    = offset
        self._bitSize   = bitSize
        self._bitOffset = bitOffset

    def get(self):
        return (self._block >> (8*self._offset+self._bitOffset))&((1<<self._bitSize)-1)

#pr.RemoteVariable replaced by TimingBitField

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
            name         = "block",
            description  = "Statistics block",
            offset       = 0x00,
            bitSize      = 0x30*8,
            bitOffset    = 0x00,
            mode         = "RO",
        ))

        self.addField(TimingBitField(    
            name         = "sofCount",
            description  = "Start of frame count",
            offset       =  0x00,
            bitSize      =  32,
            bitOffset    =  0x00,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.addField(TimingBitField(    
            name         = "eofCount",
            description  = "End of frame count",
            offset       =  0x04,
            bitSize      =  32,
            bitOffset    =  0x00,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.addField(TimingBitField(    
            name         = "FidCount",
            description  = "Valid frame count",
            offset       =  0x08,
            bitSize      =  32,
            bitOffset    =  0x00,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.addField(TimingBitField(    
            name         = "CrcErrCount",
            description  = "CRC error count",
            offset       =  0x0C,
            bitSize      =  32,
            bitOffset    =  0x00,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.addField(TimingBitField(    
            name         = "RxClkCount",
            description  = "Recovered clock count div 16",
            offset       =  0x10,
            bitSize      =  32,
            bitOffset    =  0x00,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.addField(TimingBitField(    
            name         = "RxRstCount",
            description  = "Receive link reset count",
            offset       =  0x14,
            bitSize      =  32,
            bitOffset    =  0x00,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.addField(TimingBitField(    
            name         = "RxDecErrCount",
            description  = "Receive 8b/10b decode error count",
            offset       =  0x18,
            bitSize      =  32,
            bitOffset    =  0x00,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.addField(TimingBitField(    
            name         = "RxDspErrCount",
            description  = "Receive disparity error count",
            offset       =  0x1C,
            bitSize      =  32,
            bitOffset    =  0x00,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.addField(TimingBitField(    
            name         = "RxCountReset",
            description  = "Reset receive counters",
            offset       =  0x20,
            bitSize      =  1,
            bitOffset    =  0x00,
            mode         = "WO",
            hidden       = True,
        ))

        self.addField(TimingBitField(    
            name         = "RxLinkUp",
            description  = "Receive link status",
            offset       =  0x20,
            bitSize      =  1,
            bitOffset    =  0x01,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.addField(TimingBitField(    
            name         = "RxPolarity",
            description  = "Invert receive link polarity",
            offset       =  0x20,
            bitSize      =  1,
            bitOffset    =  0x02,
            mode         = "RW",
        ))

        self.addField(TimingBitField(    
            name         = "RxReset",
            description  = "Reset receive link",
            offset       =  0x20,
            bitSize      =  1,
            bitOffset    =  0x03,
            mode         = "WO",
        ))

        self.addField(TimingBitField(    
            name         = "ClkSel",
            description  = "Select LCLS-I/LCLS-II Timing",
            offset       =  0x20,
            bitSize      =  1,
            bitOffset    =  0x04,
            mode         = "RW",
        ))

        self.addField(TimingBitField(    
            name         = "RxDown",
            description  = "Rx down latch status",
            offset       =  0x20,
            bitSize      =  1,
            bitOffset    =  0x05,
            mode         = "RW",
            verify       = False,
        ))

        self.addField(TimingBitField(    
            name         = "BypassRst",
            description  = "Buffer bypass reset status",
            offset       =  0x20,
            bitSize      =  1,
            bitOffset    =  0x06,
            mode         = "RW",
        ))

        self.addField(TimingBitField(
            name         = "RxPllReset",
            description  = "Reset RX PLL",
            offset       = 0x20,
            bitSize      = 1,
            bitOffset    = 0x07,
            mode         = "WO",
        ))

        self.addField(TimingBitField(    
            name         = "MsgDelay",
            description  = "LCLS-II timing frame pipeline delay (186MHz clks)",
            offset       =  0x24,
            bitSize      =  20,
            bitOffset    =  0x00,
            mode         = "RW",
        ))

        self.addField(TimingBitField(    
            name         = "TxClkCount",
            description  = "Transmit clock counter div 16",
            offset       =  0x28,
            bitSize      =  32,
            bitOffset    =  0x00,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.addField(TimingBitField(    
            name         = "BypassDoneCount",
            description  = "Buffer bypass done count",
            offset       =  0x2C,
            bitSize      =  16,
            bitOffset    =  0x00,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.addField(TimingBitField(    
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
            #self.RxReset.set(1)
            #time.sleep(0.001)
            #self.RxReset.set(0)    
            print('C_RxReset not implemented')

        @self.command(name="ClearRxCounters", description="Clear the Rx status counters",)
        def ClearRxCounters():
            #self.RxCountReset.set(1)
            #time.sleep(0.001)
            #self.RxCountReset.set(0)                         
            print('ClearRxCounters not implemented')

    def Dump(self):
        for k,v in self._nodes.items():
            if hasattr(v,'get'):
                print('{:} : {:}'.format(k,v.get()))

    def addField(self,node):
        setattr(self,node._name,node)
        
    def update(self):
        TimingBitField._block = self.block.get()
