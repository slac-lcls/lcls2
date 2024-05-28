#-----------------------------------------------------------------------------
# This file is part of the 'ePix HR Camera Firmware'. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the 'ePix HR Camera Firmware', including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import click
import time
import rogue
import rogue.hardware.axi
import rogue.interfaces.stream
import rogue.utilities.fileio
import os

import pyrogue as pr
import pyrogue.protocols
import pyrogue.utilities.fileio

import epix_hr_core as epixHr
import epix_uhr_dev as epixUhrDev

import numpy as np

from ePixViewer.software import *
from ePixViewer.software.deviceFiles import *
from ePixViewer.software._pseudoScope import DataReceiverPseudoScope
from ePixViewer.software._envMonitoring import DataReceiverEnvMonitoring

import subprocess

class Board(pr.Root):
    def __init__(self,
            dev                 = '/dev/datadev_0',# path to PCIe device or "sim"
            emuMode             = False,
            linkRate            = 512, # In units of Mbps
            mhzMode             = False,
            numClusters         = 14,
        **kwargs):

        self.emuMode     = emuMode
        self.linkRate    = linkRate
        self.mhzMode     = mhzMode
        self.numClusters = numClusters 
        
        # Set the timeout
        kwargs['timeout'] = 5000000 # 5.0 seconds default

        # Pass custom value to parent via super function
        super().__init__(**kwargs)

        self.dmaStream   = [[None for x in range(4)] for y in range(4)]


        ##########################################################################################################

        # Create PCIE memory mapped interface
        
        # creating the dmastreams for lane 0 VC 0 (Registers )
        self.dmaStream[0][0] = rogue.hardware.axi.AxiStreamDma(dev, (0x100*0)+0, 1) #Registers  

        ##########################################################################################################

        # Connect PGP[VC=0] to SRPv3
        self._srp = rogue.protocols.srp.SrpV3()

        self._srp == self.dmaStream[0][0]

        ##########################################################################################################

        # Add Devices
        self.add(epixHr.SysReg(
            name       = 'Core',
            memBase    = self._srp,
            offset     = 0x00000000,
            sim        = False,
            expand     = False,
            pgpVersion = 4,
            numberOfLanes = 3, #added until I fix the timing module
        ))

        self.add(epixUhrDev.App(
            name    = 'App',
            memBase = self._srp,
            offset  = 0x80000000,
            sim     = False,
            expand  = True,
        ))

        ##########################################################################################################

    def start(self,**kwargs):
        super(Board, self).start(**kwargs)

        # Get pointers to all the devices used in self.start()
        ctrl   = self.App.Ctrl
        sspMon = self.App.SspMon
        # pll    = self.App.Pll
        clkMon = self.App.ClkMon
        pll_Si5326   = self.App.Pll_Si5326

        # Refresh all the shadow variables
        self.ReadAll()

        # Print the Current Firmware Version Information
        click.secho('###############################################################################', bg='green')
        self.Core.AxiVersion.printStatus()
        click.secho('###############################################################################', bg='green')

        # # The Camera PLL is not used anymore in this project, the 371MHz is derived from the onboard clock generator
        click.secho(f'Serial Rate = {clkMon.serialRate.get()} Mb/s detected', bg='blue')

        ctrl.enable.set(True)
        sspMon.enable.set(not self.emuMode)
        self.App.WaveformControl.enable.set(True)

        # Reset all the status counters (useful for debugging with breakpoints)
        self.CountReset()
        
        # Release the ASIC global reset
        self.App.WaveformControl.GlblRstPolarity.set(0x1) # Active LOW
        
        # Board Power Cycle
        ctrl.DigPwrEn.set(0x0)
        ctrl.AnaPwrEn.set(0x0)
        time.sleep(1.0)
        ctrl.DigPwrEn.set(0x1)
        ctrl.AnaPwrEn.set(0x1)
        time.sleep(1.0)

        # Set the ASIC global reset
        click.secho('Assert AsicGlblRstL = 0x0', bg='bright_black')
        self.App.WaveformControl.GlblRstPolarity.set(0x0) # Active LOW
        time.sleep(0.5)          # Wait for things to settle

        # Release the ASIC global reset
        click.secho('Release AsicGlblRstL = 0x1', bg='bright_black')
        self.App.WaveformControl.GlblRstPolarity.set(0x1) # Active LOW
        time.sleep(1.0)            # Wait for things to settle

        # Update local RemoteVariables and verify conflagration
        self.readBlocks(recurse=True)
        self.checkBlocks(recurse=True)

        # Calibrate the ADC
        self.App.InitAdcDelay()
        # self.App.FastADCsDebug.DelayAdc0.set(55) # FW post-build calibrated value

        # Refresh all the shadow variables after status counter resets
        self.ReadAll()
        click.secho('Done')
