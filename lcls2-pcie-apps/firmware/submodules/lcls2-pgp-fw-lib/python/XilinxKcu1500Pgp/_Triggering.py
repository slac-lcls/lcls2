#!/usr/bin/env python
##############################################################################
## This file is part of 'camera-link-gen1'.
## It is subject to the license terms in the LICENSE.txt file found in the 
## top-level directory of this distribution and at: 
##    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
## No part of 'camera-link-gen1', including this file, 
## may be copied, modified, propagated, or distributed except according to 
## the terms contained in the LICENSE.txt file.
##############################################################################

import pyrogue as pr
import LclsTimingCore as timingCore

class Triggering(pr.Device):
    def __init__(   self,
            name        = "Triggering",
            description = "https://confluence.slac.stanford.edu/download/attachments/216713616/ConfigTriggeringYaml.pdf",
            numLane     = 4,
            dmaEnable   = False,
            useTap      = False,
            tickUnit    = '1/156.25MHz',
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)
           
        for i in range(numLane):
            self.add(timingCore.EvrV2ChannelReg(
                name      = f'Ch[{i}]',
                offset    = (i*0x100),
                dmaEnable = dmaEnable,
                expand    = False,
            ))

        for i in range(numLane):
            self.add(timingCore.EvrV2TriggerReg(
                name     = f'LocalTrig[{i}]',
                offset   = 0x1000 + ((i+0)*0x100),
                useTap   = useTap,
                tickUnit = tickUnit,
                expand    = False,
            ))
            
        for i in range(numLane):
            self.add(timingCore.EvrV2TriggerReg(
                name     = f'RemoteTrig[{i}]',
                offset   = 0x1000 + ((i+4)*0x100),
                useTap   = useTap,
                tickUnit = tickUnit,
                expand    = False,
            ))            
