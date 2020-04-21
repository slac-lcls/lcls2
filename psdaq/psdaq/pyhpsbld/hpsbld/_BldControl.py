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

class BldControl(pr.Device):

    def __init__(   self, 
            name        = "BldControl", 
            description = "HPS BLD Application Module", 
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        ##############################
        # Variables
        ##############################
        self.add(pr.RemoteVariable(    
            name         = 'MaxSize',
            description  = "Max output packet size in 4B words",
            offset       =  0x30000,
            bitSize      =  11,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'Activate',
            description  = "Enable output",
            offset       =  0x30000,
            bitSize      =  1,
            bitOffset    =  0x1f,
            base         = pr.UInt,
            mode         = "RW",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'ChannelMask',
            description  = "Mask of channels/variables to include",
            offset       =  0x30004,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'ChannelSevr',
            description  = "Max severity by channel",
            offset       =  0x30008,
            bitSize      =  64,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'WordsLeft',
            description  = "Words remaining in packet",
            offset       =  0x30010,
            bitSize      =  11,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'State',
            description  = "Packet state",
            offset       =  0x30010,
            bitSize      =  4,
            bitOffset    =  0x10,
            base         = pr.UInt,
            mode         = "RO",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'PulseIdL',
            description  = "Pulse Id lower word",
            offset       =  0x30014,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'TStampL',
            description  = "Timestamp lower word",
            offset       =  0x30018,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'Delta',
            description  = "Current delta timestamp",
            offset       =  0x3001c,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'PacketCount',
            description  = "Count of packets sent",
            offset       =  0x30020,
            bitSize      =  20,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'Paused',
            description  = "AxiStreamCtrl pause",
            offset       =  0x30020,
            bitSize      =  1,
            bitOffset    =  0x1f,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = 'Port',
            description  = "Host Port",
            offset       =  0x01000828,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))
        self.add(pr.RemoteVariable(    
            name         = 'Addr',
            description  = "Host IP Addr",
            offset       =  0x0100082C,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

    def Dump(self):
        for k,v in self._nodes.items():
            if hasattr(v,'get'):
                print('{:} : {:}'.format(k,v.get()))
