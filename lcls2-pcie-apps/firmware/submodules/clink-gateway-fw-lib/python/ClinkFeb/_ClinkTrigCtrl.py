#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# This file is part of the 'Camera link gateway'. It is subject to 
# the license terms in the LICENSE.txt file found in the top-level directory 
# of this distribution and at: 
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
# No part of the 'Camera link gateway', including this file, may be 
# copied, modified, propagated, or distributed except according to the terms 
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import pyrogue as pr

class ClinkTrigCtrl(pr.Device):
    def __init__(   self,       
            name        = "ClinkTrigCtrl",
            description = "Trigger Controller Container",
            **kwargs):
        super().__init__(name=name, description=description, **kwargs) 

        ##############################
        # Variables
        ##############################
        self.add(pr.RemoteVariable(    
            name         = "EnableTrig",
            description  = "Enable triggering",
            offset       = 0x000,
            bitSize      = 1,
            base         = pr.Bool,
            mode         = "RW",
        ))   

        self.add(pr.RemoteVariable(    
            name         = "InvCC",
            description  = "Inverter the 4-bit camCtrl bus",
            offset       = 0x004,
            bitSize      = 1,
            base         = pr.Bool,
            mode         = "RW",
        ))    

        self.add(pr.RemoteVariable(    
            name         = "TrigMap",
            description  = "0x0: map trigger to channel A, 0x1: map trigger to channel B",
            offset       = 0x008,
            bitSize      = 1,
            mode         = "RW",
            enum         = {
                0x0: 'ChA', 
                0x1: 'ChB', 
            },            
        ))

        self.add(pr.RemoteVariable(    
            name         = "TrigPulseWidthRaw",
            description  = "Sets the trigger pulse width on the 4-bit camCtrl bus",
            offset       = 0x00C,
            bitSize      = 16,
            mode         = "RW",
            units        = '1/125MHz',          
            hidden       = True,
        ))          
                
        self.add(pr.LinkVariable(
            name         = "TrigPulseWidth", 
            description  = "TrigPulseWidth in microseconds",
            mode         = "RW", 
            units        = "microsec",
            disp         = '{:0.3f}', 
            dependencies = [self.TrigPulseWidthRaw], 
            linkedGet    = lambda: (float(self.TrigPulseWidthRaw.value()+1) * 0.008),
            linkedSet    = lambda value, write: self.TrigPulseWidthRaw.set(int(value/0.008)-1),
        ))        
        
        self.add(pr.RemoteVariable(    
            name         = "TrigMask",
            description  = "Sets the trigger mask on the 4-bit camCtrl bus",
            offset       = 0x010,
            bitSize      = 4,
            mode         = "RW",
        ))
        
        self.add(pr.RemoteVariable(    
            name         = "TrigRate",
            description  = "Trigger Rate",
            offset       = 0x0F4,
            units        = 'Hz',
            disp         = '{:d}',
            mode         = "RO",
            pollInterval = 1,
        ))
        
        self.add(pr.RemoteVariable(    
            name         = "TrigCnt",
            description  = "Trigger Counter",
            offset       = 0x0F8,
            disp         = '{:d}',
            mode         = "RO",
            pollInterval = 1,
        ))           

        self.add(pr.RemoteVariable(
            name         = "CntRst",                 
            description  = "Counter Reset",
            mode         = 'WO',
            offset       = 0x0FC,
            hidden       = True,
        ))          
        
    def hardReset(self):
        self.CntRst.set(0x1)

    def softReset(self):
        self.CntRst.set(0x1)

    def countReset(self):
        self.CntRst.set(0x1)            
