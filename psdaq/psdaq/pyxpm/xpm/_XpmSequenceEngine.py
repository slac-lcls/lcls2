#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue XPM Sequence Engine Module
#-----------------------------------------------------------------------------
# File       : XpmSequenceEngine.py
# Created    : 2019-06-24
#-----------------------------------------------------------------------------
# Description:
# PyRogue XPM  Module
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

class SeqState(pr.Device):
    def __init__(   self, 
            name        = "SeqState", 
            description = "Sequence Engine State", 
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        self.add(pr.RemoteVariable(    
            name         = "cntReq",
            description  = "Request counts",
            offset       =  0x00,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "cntInv",
            description  = "Invalid counts",
            offset       =  0x04,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "currAddr",
            description  = "Current address",
            offset       =  0x08,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.addRemoteVariables(    
            name         = "cntCond",
            description  = "Current conditional count",
            offset       =  0x0c,
            bitSize      =  8,
            bitOffset    =  0x00,
            stride       =  1,
            base         = pr.UInt,
            mode         = "RO",
            number       = 4,
            hidden       = False,
        )

class SeqJump(pr.Device):
    def __init__(   self, 
            name        = "SeqJump", 
            description = "Sequence Jump module", 
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        self.addRemoteVariables(    
            name         = "reg",
            description  = "Register",
            offset       =  0x00,
            bitSize      =  32,
            bitOffset    =  0x00,
            stride       =  4,
            base         = pr.UInt,
            mode         = "RW",
            number       = 16,
            hidden       = True,
        )

    def setManSync(self,sync):
        r = self.reg[15].get()
        r = r & ~0xffff0000
        r = r | (0xffff0000 & (sync<<16))
        self.reg[15].set(r)

    def setManStart(self,addr,pclass):
        v = (addr&0xfff) | ((pclass&0xf)<<12)
        r = self.reg[15].get()
        r = r & ~0xffff
        r = r | (0xffff & v)
        self.reg[15].set(r)

class SeqMem(pr.Device):
    def __init__(   self, 
            name        = "SeqMem", 
            description = "Sequence memory", 
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        self.addRemoteVariables(
            name         = "mem",
            offset       = 0x00,
            bitSize      = 32,
            bitOffset    = 0x00,
            stride       = 4,
            base         = pr.UInt,
            mode         = "RW",
            number       = 2048,
            hidden       = True,
        )

class XpmSequenceEngine(pr.Device):
    def __init__(   self, 
            name        = "XpmSequenceEngine", 
            description = "XPM Sequence Engine module", 
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        self.add(pr.RemoteVariable(    
            name         = "seqAddrLen",
            description  = "Sequence address length",
            offset       =  0x00,
            bitSize      =  12,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "ctlSeq",
            description  = "Control sequences",
            offset       =  0x00,
            bitSize      =  8,
            bitOffset    =  0x10,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "xpmSeq",
            description  = "XPM sequences",
            offset       =  0x00,
            bitSize      =  8,
            bitOffset    =  0x18,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "seqEn",
            description  = "Sequence enable",
            offset       =  0x04,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "seqRestart",
            description  = "Sequence restart",
            offset       =  0x08,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            verify       = False,  # resets itself
        ))

        for i in range(16):
            self.add(SeqState(
                name     = 'SeqState_%d'%i,
                offset   = 0x80+i*0x10,
            ))

        for i in range(16):
            self.add(SeqJump(
                name     = 'SeqJump_%d'%i,
                offset   = 0x4000+i*0x40,
            ))

        for i in range(2):
            self.add(SeqMem(
                name     = 'SeqMem_%d'%i,
                offset   = 0x8000+i*0x2000,
            ))

