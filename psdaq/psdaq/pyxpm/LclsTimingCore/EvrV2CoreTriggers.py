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

from LclsTimingCore.EvrV2ChannelReg import *
from LclsTimingCore.EvrV2TriggerReg import *

class EvrV2CoreTriggers(pr.Device):
    def __init__(   self,
            name        = "EvrV2CoreTriggers",
            description = "EVR V2 Core Triggers",
            numTrig     = 16,
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        for i in range(numTrig):
            self.add(EvrV2ChannelReg(
                name   = f'EvrV2ChannelReg[{i}]',
                offset = 0x00000000 + 0x1000*i,
            ))

        for i in range(numTrig):
            self.add(EvrV2TriggerReg(
                name   = f'EvrV2TriggerReg[{i}]',
                offset = 0x00020000 + 0x1000*i,
            ))
