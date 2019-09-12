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

class Fex(pr.Device):
    def __init__(   self,       
            name        = "Fex",
            description = "Fex Container",
            **kwargs):
        super().__init__(name=name, description=description, **kwargs) 

        #pre-scaler that is being used as a placeholder for the event code filter

        self.add(batcher.AxiStreamBatcherEventBuilder( 
                name         = 'EventBuilder', 
                offset       = 0x0000, 
                numberSlaves = 2,
                tickUnit     = '156.25MHz',            
        ))
            

        self.add(tt.Prescale(
            name   = 'EVC_placeholder',
            offset = 0x1000, 
        ))

        self.add(tt.FIR( 
            offset = 0x2000, 
        ))

        self.add(tt.FrameIIR( 
            offset = 0x3000, 
        ))

        self.add(tt.Prescale(
            name   = 'background_prescaler', 
            offset = 0x6000,
        ))

        self.add(tt.FrameSubtractor( 
            offset = 0x7000, 
        ))

