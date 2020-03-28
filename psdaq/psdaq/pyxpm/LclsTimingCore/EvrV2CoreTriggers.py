#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue LCLS-II EVR V2 Core Trigger Registers
#-----------------------------------------------------------------------------
# File       : Device.py
# Created    : 2018-09-17
#-----------------------------------------------------------------------------
# Description:
# PyRogue LCLS-II EVR V2 Core Trigger Registers
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
import LclsTimingCore as timingCore

class EvrV2CoreTriggers(pr.Device):
    def __init__(   self,
            name        = "EvrV2CoreTriggers",
            description = "https://confluence.slac.stanford.edu/download/attachments/216713616/ConfigEvrV2CoreTriggersYaml.pdf",
            numTrig     = 1,
            dmaEnable   = False,
            useTap      = False,
            tickUnit    = 'TBD',
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        # Check the number of lanes requested
        if ( (numTrig<1) or (numTrig>16) ):
            raise ValueError('numTrig must be between 1 to 16: (%i) is out of range' % (numTrig) )        
        
        for i in range(numTrig):
            self.add(timingCore.EvrV2ChannelReg(
                name      = f'EvrV2ChannelReg[{i}]',
                offset    = (i*0x100),
                dmaEnable = dmaEnable,
                expand    = False,
            ))

        for i in range(numTrig):
            self.add(timingCore.EvrV2TriggerReg(
                name     = f'EvrV2TriggerReg[{i}]',
                offset   = 0x1000 + (i*0x100),
                useTap   = useTap,
                tickUnit = tickUnit,
                expand    = False,
            ))
