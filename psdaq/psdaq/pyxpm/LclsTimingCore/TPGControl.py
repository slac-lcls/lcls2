#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue Timing pattern generator control
#-----------------------------------------------------------------------------
# File       : TPGControl.py
# Created    : 2017-04-12
#-----------------------------------------------------------------------------
# Description:
# PyRogue Timing pattern generator control
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

class TPGControl(pr.Device):
    def __init__(   self,       
            name        = "TPGControl",
            description = "Timing pattern generator control",
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        ##############################
        # Variables
        ##############################

        self.add(pr.RemoteVariable(    
            name         = "NBeamSeq",
            description  = "Number of beam control sequences",
            offset       =  0x00,
            bitSize      =  8,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "NControlSeq",
            description  = "Number of experiment control sequences",
            offset       =  0x01,
            bitSize      =  8,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "NArraysBSA",
            description  = "Number of BSA arrays",
            offset       =  0x02,
            bitSize      =  8,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "SeqAddrLen",
            description  = "Sequence instruction at offset bus width",
            offset       =  0x03,
            bitSize      =  4,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "NAllowSeq",
            description  = "Number of allow table sequences",
            offset       =  0x03,
            bitSize      =  4,
            bitOffset    =  0x04,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "ClockPeriod",
            description  = "Period of beam synchronous clock (ns/cycle)",
            offset       =  0x04,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "BaseControl",
            description  = "Base rate control divisor",
            offset       =  0x08,
            bitSize      =  16,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "ACDelay",
            description  = "Adjustable delay for power line crossing measurement",
            offset       =  0x0C,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "PulseIdL",
            description  = "Pulse ID lower word",
            offset       =  0x10,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "PulseIdU",
            description  = "Pulse ID upper word",
            offset       =  0x14,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "TStampL",
            description  = "Time stamp lower word",
            offset       =  0x18,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "TStampU",
            description  = "Time stamp upper word",
            offset       =  0x1C,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.addRemoteVariables(   
            name         = "ACRateDiv",
            description  = "Power line synch rate marker divisors",
            offset       =  0x20,
            bitSize      =  8,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            number       =  6,
            stride       =  1,
        )

        self.addRemoteVariables(   
            name         = "FixedRateDiv",
            description  = "Fixed rate marker divisors",
            offset       =  0x40,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            number       =  10,
            stride       =  4,
        )

        self.add(pr.RemoteVariable(    
            name         = "RateReload",
            description  = "Loads cached ac/fixed rate marker divisors",
            offset       =  0x68,
            bitSize      =  1,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "WO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "Sync",
            description  = "Sync status with 71kHz",
            offset       =  0x70,
            bitSize      =  1,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "IrqFifoEnable",
            description  = "Enable sequence checkpoint interrupt",
            offset       =  0x74,
            bitSize      =  1,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "IrqIntvEnable",
            description  = "Enable interval counter interrupt",
            offset       =  0x74,
            bitSize      =  1,
            bitOffset    =  0x01,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "IrqBsaEnable",
            description  = "Enable BSA complete interrupt",
            offset       =  0x74,
            bitSize      =  1,
            bitOffset    =  0x02,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "IrqEnable",
            description  = "Enable interrupts",
            offset       =  0x77,
            bitSize      =  1,
            bitOffset    =  0x07,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "IrqIntvStatus",
            description  = "Interval counters updated",
            offset       =  0x78,
            bitSize      =  1,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "IrqBsaStatus",
            description  = "BSA complete updated",
            offset       =  0x78,
            bitSize      =  1,
            bitOffset    =  0x01,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "SeqFifoData",
            description  = "Sequence checkpoint data",
            offset       =  0x7C,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.addRemoteVariables(    
            name         = "BeamSeqCntl",
            description  = "Beam sequence arbitration control",
            offset       =  0x80,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            number       =  16,
            stride       =  4,
        )

        self.add(pr.RemoteVariable(    
            name         = "SeqResetL",
            description  = "Sequence restart lower word",
            offset       =  0x100,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "WO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "SeqResetU",
            description  = "Sequence restart upper word",
            offset       =  0x104,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "WO",
        ))

        self.addRemoteVariables(   
            name         = "BeamEnergy",
            description  = "Beam energy meta data",
            offset       =  0x120,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            number       =  4,
            stride       =  4,
        )

        self.add(pr.RemoteVariable(    
            name         = "BeamDiagCntl",
            description  = "Beam diagnostic buffer control",
            offset       =  0x1E4,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "WO",
        ))

        self.addRemoteVariables(   
            name         = "BeamDiagStat",
            description  = "Beam diagnostic latched status",
            offset       =  0x1E8,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
            number       =  4,
            stride       =  4,
        )

        self.add(pr.RemoteVariable(    
            name         = "BsaCompleteL",
            description  = "Bsa buffers complete lower word",
            offset       =  0x1F8,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "BsaCompleteU",
            description  = "Bsa buffers complete upper word",
            offset       =  0x1FC,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.addRemoteVariables(   
            name         = "BsaEventSel",
            description  = "Bsa definition rate/destination selection",
            offset       =  0x200,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            number       =  64,
            stride       =  8,
        )

        self.addRemoteVariables(   
            name         = "BsaStatSel",
            description  = "Bsa definition samples to average/acquire",
            offset       =  0x204,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            number       =  64,
            stride       =  8,
        )

