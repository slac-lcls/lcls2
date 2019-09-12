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

import surf.protocols.batcher as batcher
import TimeTool               as tt

class AppLane(pr.Device):
    def __init__(   self,       
            name        = "AppLane",
            description = "PCIe Application Lane Container",
            **kwargs):
        super().__init__(name=name, description=description, **kwargs) 

        self.add(batcher.AxiStreamBatcherEventBuilder( 
            name         = 'EventBuilder', 
            offset       = 0x00000, 
            numberSlaves = 3,
            tickUnit     = '156.25MHz',            
        ))
        
        self.add(tt.Fex( 
            offset = 0x10000, 
        ))

        self.add(tt.Prescale( 
            offset = 0x20000, 
        ))

        self.add(tt.ByPass( 
            offset = 0x30000, 
        ))
        
class Application(pr.Device):
    def __init__(   self,       
            name        = "Application",
            description = "PCIe Lane Container",
            numLane     = 1, # number of PGP Lanes
            **kwargs):
        super().__init__(name=name, description=description, **kwargs) 

        for i in range(numLane):
        
            self.add(AppLane(            
                name   = ('AppLane[%i]' % i), 
                offset = 0x00C00000 + (i*0x00100000), 
            ))       
