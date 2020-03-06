#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue Timing generator module for AMC Carrier
#-----------------------------------------------------------------------------
# File       : TPG.py
# Created    : 2017-04-04
#-----------------------------------------------------------------------------
# Description:
# PyRogue Timing generator module for AMC Carrier
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

from LclsTimingCore.TPGControl import *
from LclsTimingCore.TPGStatus import *
from LclsTimingCore.TPGSeqState import *
from LclsTimingCore.TPGSeqJump import *
# from LclsTimingCore.TPGSeqMem import *

class TPG(pr.Device):
    def __init__(   self, 
            name        = "TPG", 
            description = "Timing generator module for AMC Carrier", 
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        ##############################
        # Variables
        ##############################

        self.add(TPGControl(
            offset       =  0x00000000,
        ))

        self.add(TPGStatus(
            offset       =  0x00000400,
        ))

        self.add(TPGSeqState(
            offset       =  0x00000800,
        ))

        self.add(TPGSeqJump(
            offset       =  0x00000400,
        ))

        #self.add(TPGSeqMem(
        #    offset       =  0x00000400,
        #))        