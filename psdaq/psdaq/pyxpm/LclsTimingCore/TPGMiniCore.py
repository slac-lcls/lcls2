#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue Embedded timing pattern generator
#-----------------------------------------------------------------------------
# File       : TPGMiniCore.py
# Created    : 2017-04-12
#-----------------------------------------------------------------------------
# Description:
# PyRogue Embedded timing pattern generator
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

class TPGMiniCore(pr.Device):
    def __init__(   self,       
            name        = "TPGMiniCore",
            description = "Embedded timing pattern generator",
            NARRAYSBSA  = 2,
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)
        ##############################
        # Variables
        ##############################

        self.add(pr.RemoteVariable(    
            name         = "TxPolarity",
            description  = "Invert transmit link polarity",
            offset       = 0x00,
            bitSize      = 1,
            bitOffset    = 1,
            base         = pr.UInt,
            mode         = "RW",
        ))
        
        self.add(pr.RemoteVariable(    
            name         = "TxLoopback",
            description  = "Set transmit link loopback",
            offset       = 0x00,
            bitSize      = 3,
            bitOffset    = 2,
            base         = pr.UInt,
            mode         = "RW",
        ))
        
        self.add(pr.RemoteVariable(    
            name         = "TxInhibit",
            description  = "Set transmit link inhibit",
            offset       = 0x00,
            bitSize      = 1,
            bitOffset    = 5,
            base         = pr.UInt,
            mode         = "RW",
        ))
        
        self.add(pr.RemoteVariable(    
            name         = "BaseControl",
            description  = "Base rate trigger divisor",
            offset       = 0x04,
            bitSize      = 16,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "RW",
        ))
        
        self.add(pr.RemoteVariable(    
            name         = "PulseIdWr",
            description  = "Pulse ID write",
            offset       = 0x58,
            bitSize      = 64,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "WO",
        ))
        
        self.add(pr.RemoteVariable(    
            name         = "PulseIdRd",
            description  = "Pulse ID read",
            offset       = 0x08,
            bitSize      = 64,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "RO",
        ))        
        
        self.add(pr.RemoteVariable(    
            name         = "PulseIdSet",
            description  = "Activates PulseId register value",
            offset       = 0x70,
            bitSize      = 1,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "RW",
        ))        
        
        self.add(pr.RemoteVariable(    
            name         = "TStampWr",
            description  = "Time stamp Write",
            offset       = 0x60,
            bitSize      = 64,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "WO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "TStampRd",
            description  = "Time stamp read",
            offset       = 0x10,
            bitSize      = 64,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "RO",
        ))        
        
        self.add(pr.RemoteVariable(    
            name         = "TStampSet",
            description  = "Activates Timestamp register value",
            offset       = 0x74,
            bitSize      = 1,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "RW",
        ))        
                
        self.addRemoteVariables(    
            name         = "FixedRateDiv",
            description  = "Fixed rate marker divisors",
            offset       = 0x18,
            bitSize      = 32,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "RW",
            number       = 10,
            stride       = 4,
            hidden       = True,
        )

        self.add(pr.RemoteVariable(    
            name         = "RateReload",
            description  = "Loads cached fixed rate marker divisors",
            offset       = 0x40,
            bitSize      = 1,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "WO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "NBeamSeq",
            description  = "Number of beam request engines",
            offset       = 0x4C,
            bitSize      = 6,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "NControlSeq",
            description  = "Number of control sequence engines",
            offset       = 0x4C,
            bitSize      = 8,
            bitOffset    = 6,
            base         = pr.UInt,
            mode         = "RO",
        ))        
        
        self.add(pr.RemoteVariable(    
            name         = "NArraysBsa",
            description  = "Number of BSA arrays",
            offset       = 0x4C,
            bitSize      = 8,
            bitOffset    = 14,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "SeqAddrLen",
            description  = "Number of beam sequence engines",
            offset       = 0x4C,
            bitSize      = 4,
            bitOffset    = 22,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "NAllowSeq",
            description  = "Number of beam allow engines",
            offset       = 0x4C,
            bitSize      = 6,
            bitOffset    = 26,
            base         = pr.UInt,
            mode         = "RO",
        ))        
        
        self.add(pr.RemoteVariable(    
            name         = "TxReset",
            description  = "Reset transmit link",
            offset       = 0x68,
            bitSize      = 1,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "WO",
        )) 

        self.add(pr.RemoteVariable(    
            name         = "CountIntervalReset",
            description  = "Count Interval Reset",
            offset       = 0x6C,
            bitSize      = 1,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "WO",
        ))         
        
        self.add(pr.RemoteVariable(    
            name         = "Lcls1BsaNumSamples",
            description  = "Lcls-1 BSA Number of Samples - 1",
            offset       = 0x78,
            bitSize      = 12,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "RW",
        ))  

        self.add(pr.RemoteVariable(   
            name         = "Lcls1BsaRate",
            description  = "Lcls-1 BSA Rate",
            offset       = 0x78,
            bitSize      = 3,
            bitOffset    = 12,
            mode         = "RW",
            enum         = {
                0 : "120Hz",
                1 : "60Hz",
                2 : "30Hz",
                3 : "10Hz",
                4 : "5Hz",
                5 : "1Hz",
                6 : "0.5Hz",
            },
        ))
        
        self.add(pr.RemoteVariable(   
            name         = "Lcls1BsaTimeSlot",
            description  = "Lcls-1 BSA Time Slot",
            offset       = 0x78,
            bitSize      = 3,
            bitOffset    = 15,
            mode         = "RW",
            enum         = {
                0 : "TS1",
                1 : "TS2",
                2 : "TS3",
                3 : "TS4",
                4 : "TS5",
                5 : "TS6",
            },
        )) 

        self.add(pr.RemoteVariable(   
            name         = "Lcls1BsaSeverity",
            description  = "Lcls-1 BSA Rejection Severity Threshold",
            offset       = 0x78,
            bitSize      = 2,
            bitOffset    = 18,
            mode         = "RW",
            enum         = {
                0 : "INVALID",
                1 : "MAJOR",
                2 : "MINOR",
            },
        ))                
        
        self.add(pr.RemoteVariable(    
            name         = "Lcls1BsaEdefSlot",
            description  = "Lcls-1 BSA EDEF Slot Number",
            offset       = 0x78,
            bitSize      = 4,
            bitOffset    = 20,
            base         = pr.UInt,
            mode         = "RW",
        ))  

        self.add(pr.RemoteVariable(    
            name         = "Lcls1BsaNumAvgs",
            description  = "Lcls-1 BSA Number of Values to Average per Sample - 1",
            offset       = 0x78,
            bitSize      = 8,
            bitOffset    = 24,
            base         = pr.UInt,
            mode         = "RW",
        ))         
        
        self.add(pr.RemoteVariable(    
            name         = "Lcls1BsaStart",
            description  = "Lcls-1 BSA Started by Writing any Value Here",
            offset       = 0x7C,
            bitSize      = 32,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "RW",
        ))             

        self.add(pr.RemoteVariable(    
            name         = "BsaCompleteWr",
            description  = "BSA complete write",
            offset       = 0x50,
            bitSize      = 64,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "WO",
            overlapEn    = True,            
        ))

        self.add(pr.RemoteVariable(    
            name         = "BsaCompleteRd",
            description  = "BSA complete read",
            offset       = 0x50,
            bitSize      = 64,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "RO",
            overlapEn    = True,            
        ))
        
        self.addRemoteVariables(  
            name         = "BsaActive",
            description  = "Activates/Deactivates BSA EDEF",
            offset       = 0x01FC,
            bitSize      = 1,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "RW",
            number       = NARRAYSBSA,
            stride       = 1,
            hidden       = True,
        )        

        self.addRemoteVariables(  
            name         = "BsaRateSelMode",
            description  = "BSA def rate mode selection",
            offset       = 0x200,
            bitSize      = 2,
            bitOffset    = 0,
            mode         = "RW",
            number       = NARRAYSBSA,
            stride       = 16,
            hidden       = True,
            overlapEn    = True,            
            enum         = {
                0 : "FixedRate",
                1 : "ACRate",
                2 : "Sequencer",
            },    
        )
        
        self.addRemoteVariables(  
            name         = "BsaFixedRate",
            description  = "BSA fixed rate mode selection",
            offset       = 0x200,
            bitSize      = 4,
            bitOffset    = 2,
            mode         = "RW",
            number       = NARRAYSBSA,
            stride       = 16,
            hidden       = True,
            overlapEn    = True,            
            enum         = {
                0 : "1MHz",
                1 : "71kHz",
                2 : "10kHz",
                3 : "1kHz",
                4 : "100Hz",
                5 : "10Hz",
                6 : "1Hz",
            },            
        ) 
        
        self.addRemoteVariables(  
            name         = "BsaACRate",
            description  = "BSA AC rate mode selection",
            offset       = 0x200,
            bitSize      = 3,
            bitOffset    = 6,
            mode         = "RW",
            number       = NARRAYSBSA,
            stride       = 16,
            hidden       = True,
            overlapEn    = True,            
            enum         = {
                0 : "60Hz",
                1 : "30Hz",
                2 : "10Hz",
                3 : "5Hz",
                4 : "1Hz",
                5 : "0.5Hz",
            },            
        )         
        
        self.addRemoteVariables(  
            name         = "BsaACTSMask",
            description  = "BSA AC timeslot mask selection",
            offset       = 0x200,
            bitSize      = 6,
            bitOffset    = 9,
            base         = pr.UInt,
            mode         = "RW",
            number       = NARRAYSBSA,
            stride       = 16,
            hidden       = True,
            overlapEn    = True,            
        ) 

        self.addRemoteVariables(  
            name         = "BsaSequenceSelect",
            description  = "BSA sequencer selection",
            offset       = 0x200,
            bitSize      = 5,
            bitOffset    = 15,
            base         = pr.UInt,
            mode         = "RW",
            number       = NARRAYSBSA,
            stride       = 16,
            hidden       = True,
            overlapEn    = True,            
        )    

        self.addRemoteVariables(  
            name         = "BsaSequenceBitSelect",
            description  = "BSA sequencer bit selection",
            offset       = 0x200,
            bitSize      = 4,
            bitOffset    = 20,
            base         = pr.UInt,
            mode         = "RW",
            number       = NARRAYSBSA,
            stride       = 16,
            hidden       = True,
            overlapEn    = True,            
        )            
        
        self.addRemoteVariables(  
            name         = "BsaDestMode",
            description  = "BSA destination mode",
            offset       = 0x200,
            bitSize      = 2,
            bitOffset    = 24,
            mode         = "RW",
            number       = NARRAYSBSA,
            stride       = 16,
            hidden       = True,
            overlapEn    = True,            
            enum         = {
                0 : "Dont_Care",
                1 : "Inclusive",
                2 : "Exclusive",
            },            
        )           
        
        self.addRemoteVariables(  
            name         = "BsaDestInclusiveMask",
            description  = "BSA inclusive destination mask",
            offset       = 0x204,
            bitSize      = 16,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "RW",
            number       = NARRAYSBSA,
            stride       = 16,
            hidden       = True,
            overlapEn    = True,            
        )      

        self.addRemoteVariables(  
            name         = "BsaDestExclusiveMask",
            description  = "BSA exclusive destination mask",
            offset       = 0x204,
            bitSize      = 16,
            bitOffset    = 16,
            base         = pr.UInt,
            mode         = "RW",
            number       = NARRAYSBSA,
            stride       = 16,
            hidden       = True,
            overlapEn    = True,            
        )              

        self.addRemoteVariables(    
            name         = "BsaNtoAvg",
            description  = "BSA def num acquisitions to average",
            offset       = 0x208,
            bitSize      = 13,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "RW",
            number       = NARRAYSBSA,
            stride       = 16,
            hidden       = True,
            overlapEn    = True,            
        )

        self.addRemoteVariables(   
            name         = "BsaAvgToWr",
            description  = "BSA def num averages to record",
            offset       = 0x208,
            bitSize      = 16,
            bitOffset    = 16,
            base         = pr.UInt,
            mode         = "RW",
            number       = NARRAYSBSA,
            stride       = 8,
            hidden       = True,
            overlapEn    = True,            
        )
        
        self.addRemoteVariables(  
            name         = "BsaMaxSeverity",
            description  = "BSA def max alarm severity",
            offset       = 0x208,
            bitSize      = 2,
            bitOffset    = 14,
            mode         = "RW",
            number       = NARRAYSBSA,
            stride       = 16,
            hidden       = True,
            overlapEn    = True,            
            enum         = {
                0 : "NoAlarm",
                1 : "Minor",
                2 : "Major",
                3 : "Invalid",
            },            
        ) 

        self.add(pr.RemoteVariable(    
            name         = "PllCnt",
            description  = "Count of PLL status changes",
            offset       = 0x500,
            bitSize      = 32,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "ClkCnt",
            description  = "Count of local 186M clock",
            offset       = 0x504,
            bitSize      = 32,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "SyncErrCnt",
            description  = "Count of 71k sync errors",
            offset       = 0x508,
            bitSize      = 32,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "CountInterval",
            description  = "Interval counters update period",
            offset       = 0x50C,
            bitSize      = 32,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "BaseRateCount",
            description  = "Count of base rate triggers",
            offset       = 0x510,
            bitSize      = 32,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "RO",
        ))
