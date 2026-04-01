#!/usr/bin/env python3
##############################################################################
## This file is part of 'EPIX'.
## It is subject to the license terms in the LICENSE.txt file found in the 
## top-level directory of this distribution and at: 
##    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
## No part of 'EPIX', including this file, 
## may be copied, modified, propagated, or distributed except according to 
## the terms contained in the LICENSE.txt file.
##############################################################################

import surf.devices.silabs as silabs
import pyrogue as pr
import os
import time

class Si5394(silabs.Si5394):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        self.add(pr.LocalCommand(name='program',
                                 description='[119MHz, 185.7MHz]',
                                 function=self._program))

    def _program(self,clkSel=1):
        self.enable.set(True)
        self._programClk([clkSel])
        self.enable.set(False)

    def _programClk(self,args):
        path = os.path.dirname(os.path.abspath(__file__))+'/kcu/config/Si5394-186MHz.csv'
        print(f'program {path}')
        self.LoadCsvFile( path )
