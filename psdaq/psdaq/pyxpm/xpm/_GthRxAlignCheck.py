#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue Timing frame phase lock
#-----------------------------------------------------------------------------
# File       : GthRxAlignCheck.py
# Created    : 2017-04-12
#-----------------------------------------------------------------------------
# Description:
# PyRogue Timing frame phase lock
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

class GthRxAlignCheck(pr.Device):
    def __init__(   self,       
            name        = "GthRxAlignCheck",
            description = "Timing frame phase lock",
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        ##############################
        # Variables
        ##############################

        # self.addRemoteVariables(   
            # name         = "PhaseCount",
            # description  = "Timing frame phase",
            # offset       =  0x00,
            # bitSize      =  16,
            # bitOffset    =  0x00,
            # mode         = "RO",
            # pollInterval = 1,
            # number       =  128,
            # stride       =  2,
            # hidden       =  True,
        # )
                        
        self.addRemoteVariables(   
            name         = "PhaseCount",
            description  = "Timing frame phase",
            offset       =  0x00,
            bitSize      =  32,
            bitOffset    =  0x00,
            mode         = "RO",
#  Too many accesses?
#            pollInterval = 1,
            number       =  64,
            stride       =  4,
            hidden       =  True,
        )                       

        self.add(pr.RemoteVariable(    
            name         = "PhaseTarget",
            description  = "Timing frame phase lock target",
            offset       =  0x100,
            bitSize      =  7,
            bitOffset    =  0x00,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "PhaseTargetMask",
            description  = "Timing frame phase lock target",
            offset       =  0x100,
            bitSize      =  7,
            bitOffset    =  0x08,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "ResetLen",
            description  = "Reset length",
            offset       =  0x100,
            bitSize      =  4,
            bitOffset    =  16,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "LastPhase",
            description  = "Last timing frame phase seen",
            offset       =  0x104,
            bitSize      =  7,
            bitOffset    =  0x00,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.add(pr.RemoteVariable(
            name         = "TxClkFreqRaw",
            offset       = 0x108, 
            bitSize      = 32, 
            bitOffset    = 0, 
            mode         = 'RO', 
            hidden       = True,
            pollInterval = 1,
        ))

        self.add(pr.LinkVariable(
            name         = "TxClkFreq",
            units        = "MHz",
            mode         = 'RO',
            dependencies = [self.TxClkFreqRaw], 
            linkedGet    = lambda: self.TxClkFreqRaw.value() * 1.0e-6,
            disp         = '{:0.3f}',
        ))

        self.add(pr.RemoteVariable(
            name         = "RxClkFreqRaw",
            offset       = 0x10C, 
            bitSize      = 32, 
            bitOffset    = 0, 
            mode         = 'RO', 
            hidden       = True,
            pollInterval = 1,
        ))

        self.add(pr.LinkVariable(
            name         = "RxClkFreq",
            units        = "MHz",
            mode         = 'RO',
            dependencies = [self.RxClkFreqRaw], 
            linkedGet    = lambda: self.RxClkFreqRaw.value() * 1.0e-6,
            disp         = '{:0.3f}',
        ))        

        self.addRemoteVariables(0x300,0x4,
                                name    = 'Drp',
                                offset  = 0x4000,
        )

        if False:
            self.add(pr.RemoteVariable(
                name         = "RxLockState",
                offset       = 0x180,
                bitSize      = 32,
                mode         = 'RO',
                hidden       = True,
            ))

            self.add(pr.RemoteVariable(
                name         = "ResetState",
                offset       = 0x184,
                bitSize      = 32,
                mode         = 'RO',
                hidden       = True,
            ))
            
            self.add(pr.RemoteVariable(
                name         = "RxCtrl0Out",
                offset       = 0x188,
                bitSize      = 16,
                mode         = 'RO',
                hidden       = True,
            ))

            self.add(pr.RemoteVariable(
                name         = "RxCtrl1Out",
                offset       = 0x18C,
                bitSize      = 16,
                mode         = 'RO',
                hidden       = True,
            ))

            self.add(pr.RemoteVariable(
                name         = "RxCtrl3Out",
                offset       = 0x190,
                bitSize      = 8,
                mode         = 'RO',
                hidden       = True,
            ))

    def dump(self):
        def print_field(name,bitoff,bitmask,v=v):
            print(f'{name:20.20s}: {(v>>bitoff)&bitmask}')

        v = 0xffffffff
        if False:
            v = self.RxLockState.get()

        print_field("gth_rxalign_resetIn"  , 0, 1)
        print_field("gth_rxalign_resetDone", 1, 1)
        print_field("gth_rxalign_resetErr" , 2, 1)
        print_field("gth_rxalign_r_locked" , 3, 1)
        print_field("gth_rxalign_r_rst"    , 4, 1)
        print_field("gth_rxalign_r_rstcnt" , 8, 0xf)
        print_field("gth_rxalign_r_state"  , 12, 3)

        if False:
            v = self.ResetState.get()

        print_field("gth_txstatus_clkactive", 0, 1)
        print_field("gth_txstatus_bypassrst", 1, 1)
        print_field("gth_txstatus_pllreset" , 2, 1)
        print_field("gth_txstatus_datareset", 3, 1)
        print_field("gth_txstatus_resetdone", 4, 1)

        print_field("gth_rxstatus_clkactive",16, 1)
        print_field("gth_rxstatus_bypassrst",17, 1)
        print_field("gth_rxstatus_pllreset" ,18, 1)
        print_field("gth_rxstatus_datareset",19, 1)
        print_field("gth_rxstatus_resetdone",20, 1)
        print_field("gth_rxstatus_bypassdon",21, 1)
        print_field("gth_rxstatus_bypasserr",22, 1)
        print_field("gth_rxstatus_cdrstable",23, 1)

        print_field("gth_rxctrl0out",0,0xffff,self.RxCtrl0Out.get())
        print_field("gth_rxctrl1out",0,0xffff,self.RxCtrl1Out.get())
        print_field("gth_rxctrl3out",0,0xff  ,self.RxCtrl3Out.get())

        for i in range(0,256):
            print(f'drp_{i:02x} : {self.Drp[i].get()&0xffff:04x}  {self.Drp[i+256].get()&0xffff:04x}  {self.Drp[i+512].get()&0xffff:04x}')
