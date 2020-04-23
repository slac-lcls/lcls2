#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# This file is part of the 'L2S-I DAQ'. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the 'Camera link gateway', including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
import pyrogue as pr
import rogue
import click
import psdaq.pyhpsbld.hpsbld as hps

rogue.Version.minVersion('4.9.0')

class HpsRoot(pr.Root):

    def __init__(self,
                 ipAddr      = '192.168.0.1',
                 **kwargs):

        # Pass custom value to parent via super function
        super().__init__(**kwargs)

        # Set base
        self.add(hps.Top(
            ipAddr = ipAddr
        ))
    
