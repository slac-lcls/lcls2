#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue TDM SFP diagnostics
#-----------------------------------------------------------------------------
# File       : MpsSfpAmc.py
# Created    : 2019-06-24
#-----------------------------------------------------------------------------
# Description:
# PyRogue SFP Diagnostics Module
#-----------------------------------------------------------------------------
# This file is part of the rogue software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the rogue software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import pyrogue                 as pr
import psdaq.pyxpm.xpm         as xpm
from surf.devices.transceivers import Sff8472

class MpsSfpAmc(pr.Device):

    def __init__(   self, 
                    name        = "MpsSfpAmc", 
                    description = "SFP Diagnostics Module", 
                    **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        self.add(xpm.SfpSummary(
#            memBase = self.srp,
            name        = 'SfpSummary',
            description = 'PCA9506',
            offset      = 0,
        ))

        self.add(pr.RemoteVariable(
            name        = 'I2cMux',
            description = 'PCA9547 I2C Mux Control',
            offset      = 0x400,
            bitSize     = 8,
            bitOffset   = 0,
            base        = pr.UInt,
            mode        = "RW",
        ))

        self.add(xpm.SfpI2c(
#            memBase = self.srp,
            name    = 'SfpI2c',
            offset  = 0x800,
        ))
