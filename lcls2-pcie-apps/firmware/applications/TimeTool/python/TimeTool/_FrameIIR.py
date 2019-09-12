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

class FrameIIR(pr.Device):
    def __init__(   self,       
            name        = "FrameIIR",
            description = "Frame Infinite Impulse Response",
            **kwargs):
        super().__init__(name=name, description=description, **kwargs) 
        
        self.add(pr.RemoteVariable(   
            name         = 'ScratchPad',
            description  = 'Register to test reads and writes',
            offset       = 0x0000,
            bitSize      = 32,
            bitOffset    = 0,
            mode         = 'RW',
            disp         = '{:#08x}',
        ))        

        self.add(pr.RemoteVariable(    
            name         = "TimeConstant",
            description  = 'Measured in units of 2**x',
            offset       =  0x0004,
            bitSize      =  32,
            bitOffset    =  0,
            mode         = "RW",
        ))
