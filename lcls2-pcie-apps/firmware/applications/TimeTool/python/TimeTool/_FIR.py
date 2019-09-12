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

class FIR(pr.Device):
    def __init__(   self,       
            name        = "FIR",
            description = "Finite Impulse Response",
            **kwargs):
        super().__init__(name=name, description=description, **kwargs) 

        for i in range(8):
            self.add(pr.RemoteVariable(   
                name         = 'CoefficientSet'+str(i),
                description  = 'FIR coefficients',
                offset       = i*4,
                bitSize      = 32,
                bitOffset    = 0,
                mode         = 'RW',
                disp         = '{:#08x}',
            ))        
        self.add(pr.RemoteVariable(   
            name         = 'LoadCoefficients',
            description  = 'LoadCoefficients',
            offset       = 32,
            bitSize      = 1,
            bitOffset    = 0,
            mode         = 'RW',
            verify       = False, 
            disp         = '{:#01x}',
        ))        
