#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue XPM Timing Delay Application Module
#-----------------------------------------------------------------------------
# File       : TimingDelayAppl.py
# Created    : 2019-06-24
#-----------------------------------------------------------------------------
# Description:
# PyRogue Timing Delay Application Module
#-----------------------------------------------------------------------------
# This file is part of the rogue software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the rogue software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import pyrogue        as pr
import psdaq.pyhpsbld.hpsbld        as hps

class AmcCarrierBsa(pr.Device):

    def __init__(   self, 
            name        = "AmcCarrierBsa", 
            description = "HPS BLD Application Module", 
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        self.add(hps.BldAxiStream(
            name     = 'BsssControl',
            offset   = 0x00030000,
            numEdefs = 9,
        ))

        self.add(hps.BldAxiStream(
            name     = 'BldControl',
            offset   = 0x00040000,
            numEdefs = 4,
        ))

        self.add(hps.BsasControl(
            name    = 'BsasControl',
            offset  = 0x00050000,
        ))
