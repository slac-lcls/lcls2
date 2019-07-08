#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue High Speed Repeater Module
#-----------------------------------------------------------------------------
# File       : HSRepeater.py
# Created    : 2019-06-24
#-----------------------------------------------------------------------------
# Description:
# PyRogue High Speed Repeater Module
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

class Ds125br401Channel(pr.Device):
    def __init__(   self, 
            name        = "Ds125br401_Channel", 
            description = "DS125BR401 Channel", 
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        ##############################
        # Variables
        ##############################

        self.add(pr.RemoteVariable(    
            name         = "sigdetPreset",
            description  = "Signal detector force on",
            offset       =  0x00,
            bitSize      =  1,
            bitOffset    =  0x01,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "sigdetReset",
            description  = "Signal detector force off",
            offset       =  0x00,
            bitSize      =  1,
            bitOffset    =  0x02,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "rxDet",
            description  = "Receive detect",
            offset       =  0x04,
            bitSize      =  2,
            bitOffset    =  0x02,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "idleSel",
            description  = "Idle select",
            offset       =  0x04,
            bitSize      =  1,
            bitOffset    =  0x04,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "idleAuto",
            description  = "Idle auto",
            offset       =  0x04,
            bitSize      =  1,
            bitOffset    =  0x05,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "eqCtl",
            description  = "Equalization control",
            offset       =  0x08,
            bitSize      =  8,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "vodCtl",
            description  = "VOD control",
            offset       =  0x0c,
            bitSize      =  3,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "modeSel",
            description  = "Mode select",
            offset       =  0x0c,
            bitSize      =  1,
            bitOffset    =  0x06,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "shortCircProt",
            description  = "Short circuit protection",
            offset       =  0x0c,
            bitSize      =  1,
            bitOffset    =  0x07,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "demCtl",
            description  = "Deemphasis control",
            offset       =  0x10,
            bitSize      =  3,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "modDetStatus",
            description  = "Mode detect status",
            offset       =  0x10,
            bitSize      =  2,
            bitOffset    =  0x05,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "rxDetStatus",
            description  = "Rx detect status",
            offset       =  0x10,
            bitSize      =  1,
            bitOffset    =  0x07,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "idleThd",
            description  = "Idle deassert threshold",
            offset       =  0x14,
            bitSize      =  2,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "idleTha",
            description  = "Idle assert threshold",
            offset       =  0x14,
            bitSize      =  2,
            bitOffset    =  0x02,
            base         = pr.UInt,
            mode         = "RW",
        ))

        @self.command(name="Init", description="Initialize channel",)
        def Init():
            self.sigdetPreset.set(1)
            self.modeSel.set(0)
            self.eqCtl.set(0)

class Ds125br401(pr.Device):
    def __init__(   self, 
            name        = "HSRepeater", 
            description = "XPM High Speed Repeater Module", 
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        ##############################
        # Variables
        ##############################

        self.add(pr.RemoteVariable(    
            name         = "eepromRdDn",
            description  = "EEPROM read done",
            offset       =  0x00,
            bitSize      =  1,
            bitOffset    =  0x02,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "addr",
            description  = "Address bits",
            offset       =  0x00,
            bitSize      =  4,
            bitOffset    =  0x03,
            base         = pr.UInt,
            mode         = "RO",
        ))

        for i in range(8):
            self.add(pr.RemoteVariable(    
                name         = "pwdnChan_%d"%i,
                description  = "Power down channel %d"%i,
                offset       =  0x04,
                bitSize      =  1,
                bitOffset    =  i,
                base         = pr.UInt,
                mode         = "RW",
            ))

        self.add(pr.RemoteVariable(    
            name         = "pwdnOverride",
            description  = "Power down override",
            offset       =  0x08,
            bitSize      =  1,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "loopbCtl",
            description  = "Loopback control",
            offset       =  0x08,
            bitSize      =  2,
            bitOffset    =  0x04,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "smbusEn",
            description  = "SMBUS enable",
            offset       =  0x18,
            bitSize      =  1,
            bitOffset    =  0x03,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "smbusRst",
            description  = "Reset SMBUS",
            offset       =  0x1c,
            bitSize      =  1,
            bitOffset    =  0x05,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "regRst",
            description  = "Reset registers",
            offset       =  0x1c,
            bitSize      =  1,
            bitOffset    =  0x06,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "modePin",
            description  = "Override mode",
            offset       =  0x20,
            bitSize      =  1,
            bitOffset    =  0x02,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "rxdetPin",
            description  = "Override rxdet",
            offset       =  0x20,
            bitSize      =  1,
            bitOffset    =  0x03,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "idlePin",
            description  = "Override idle",
            offset       =  0x20,
            bitSize      =  1,
            bitOffset    =  0x04,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "sigdetThPin",
            description  = "Override signal detect threshold",
            offset       =  0x20,
            bitSize      =  1,
            bitOffset    =  0x06,
            base         = pr.UInt,
            mode         = "RW",
        ))

        for i in range(8):
            self.add(pr.RemoteVariable(    
                name         = "sigdetMon_%d"%i,
                description  = "Signal detect threshold status %d"%i,
                offset       =  0x28,
                bitSize      =  1,
                bitOffset    =  i,
                base         = pr.UInt,
                mode         = "RW",
            ))

#        for i in range(4):
#            self.add(Ds125br401Channel(
#                name     = "ChannelB_%d"%i,
#                offset   = 0x30+0x18*i,
#            ))

        self.add(pr.RemoteVariable(    
            name         = "redSdGain",
            description  = "Reduced signal detect gain",
            offset       =  0xa0,
            bitSize      =  2,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "fastIdle",
            description  = "Signal detect fast idle",
            offset       =  0xa0,
            bitSize      =  2,
            bitOffset    =  0x02,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "highIdle",
            description  = "Signal detect high idle",
            offset       =  0xa0,
            bitSize      =  2,
            bitOffset    =  0x04,
            base         = pr.UInt,
            mode         = "RW",
        ))

#        for i in range(4):
#            self.add(Ds125br401Channel(
#                name     = "ChannelA_%d"%i,
#                offset   = 0xa4+0x18*i,
#            ))

        self.add(pr.RemoteVariable(    
            name         = "devVer",
            description  = "Device version",
            offset       =  0x144,
            bitSize      =  5,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "devId",
            description  = "Device id",
            offset       =  0x144,
            bitSize      =  3,
            bitOffset    =  0x05,
            base         = pr.UInt,
            mode         = "RO",
        ))

        @self.command(name="Init", description="Initialize bus",)
        def Init():
            self.smbusEn.set(1)
            self.channelA_0.Init()
            self.channelB_0.Init()
            self.channelA_1.Init()
            self.channelB_1.Init()
            self.channelA_2.Init()
            self.channelB_2.Init()
            self.channelA_3.Init()
            self.channelB_3.Init()
                

