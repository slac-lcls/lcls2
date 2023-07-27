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

class BldEdef(pr.Device):

    def __init__(   self, 
                    name        = "BldEdef", 
                    description = "HPS BLD Application Module", 
                    numEdefs    = 4,
                    **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        ##############################
        # Variables
        ##############################
        self.add(pr.RemoteVariable(    
            name         = 'rateSel',
            description  = "Rate select",
            offset       =  0x0,
            bitSize      =  13,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'destSel',
            description  = "Destn select",
            offset       =  0x0,
            bitSize      =  19,
            bitOffset    =  0x0d,
            base         = pr.UInt,
            mode         = "RW",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'tsUpdate',
            description  = "Timestamp bit update",
            offset       =  0x4,
            bitSize      =  5,
            bitOffset    =  0x0,
            base         = pr.UInt,
            mode         = "RW",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'Enable',
            description  = "Enable",
            offset       =  0x4,
            bitSize      =  1,
            bitOffset    =  0x1f,
            base         = pr.UInt,
            mode         = "RW",
        ))

class BldAxiStream(pr.Device):

    def __init__(   self, 
                    name        = "BldAxiStream", 
                    description = "HPS BLD Application Module", 
                    numEdefs    = 4,
                    **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        ##############################
        # Variables
        ##############################
        self.add(pr.RemoteVariable(    
            name         = 'packetSize',
            description  = "Max output packet size in 4B words",
            offset       =  0x0,
            bitSize      =  11,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'Enable',
            description  = "Enable output",
            offset       =  0x0,
            bitSize      =  1,
            bitOffset    =  0x1f,
            base         = pr.UInt,
            mode         = "RW",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'channelMask',
            description  = "Mask of channels/variables to include",
            offset       =  0x04,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'count',
            description  = "Words remaining in packet",
            offset       =  0x10,
            bitSize      =  11,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'state',
            description  = "Packet state",
            offset       =  0x10,
            bitSize      =  4,
            bitOffset    =  0x10,
            base         = pr.UInt,
            mode         = "RO",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'pulseIdL',
            description  = "Pulse Id lower word",
            offset       =  0x14,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'tStampL',
            description  = "Timestamp lower word",
            offset       =  0x18,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'delta',
            description  = "Current delta timestamp",
            offset       =  0x1c,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'packets',
            description  = "Count of packets sent",
            offset       =  0x20,
            bitSize      =  20,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'paused',
            description  = "AxiStreamCtrl pause",
            offset       =  0x20,
            bitSize      =  1,
            bitOffset    =  0x1f,
            base         = pr.UInt,
            mode         = "RO",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'depth',
            description  = "FIFO depth",
            offset       =  0x24,
            bitSize      =  32,
            bitOffset    =  0,
            base         = pr.UInt,
            mode         = "RO",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'diagnClkFreq',
            description  = "Diagnostic Clk Freq",
            offset       =  0x28,
            bitSize      =  29,
            bitOffset    =  0x0,
            base         = pr.UInt,
            mode         = "RO",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'diagnStrobeRate',
            description  = "Diagnostic Strobe Rate",
            offset       =  0x2c,
            bitSize      =  29,
            bitOffset    =  0x0,
            base         = pr.UInt,
            mode         = "RO",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'eventSel0Rate',
            description  = "Event select rate",
            offset       =  0x30,
            bitSize      =  32,
            bitOffset    =  0x0,
            base         = pr.UInt,
            mode         = "RO",
        ))
        for i in range(numEdefs):
            self.add(BldEdef(
                name        = f'Edef{i}',
                description = 'Bld EDEF',
                offset      = 0x40+i*8,
            ))

    def Dump(self):
        for k,v in self._nodes.items():
            if hasattr(v,'get'):
                print('{:} : {:}'.format(k,v.get()))
