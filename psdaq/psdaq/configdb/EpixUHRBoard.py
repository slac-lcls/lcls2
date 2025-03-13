#-----------------------------------------------------------------------------
# This file is part of the 'epix-uhr-100kHz-dev'. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the 'epix-uhr-100kHz-dev', including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import pyrogue as pr
import pyrogue.protocols
import pyrogue.utilities.fileio
import pyrogue.interfaces.simulation

import rogue
import rogue.hardware.axi
import rogue.interfaces.stream
import rogue.utilities.fileio

from psdaq.utils import enable epix_uhr_gtreadout_dev
import epix_uhr_gtreadout_dev as epixUhrDev
import epix_hr_leap_common as leapCommon

import os
import numpy as np
import time
import click
import subprocess

rogue.Version.minVersion('6.0.0')

try :
    from ePixViewer.asics import ePixUhr100kHz
    from ePixViewer import EnvDataReceiver
    from ePixViewer import ScopeDataReceiver
except ImportError:
    pass

class Root(pr.Root):
    def __init__(self,
        dev                 = '/dev/datadev_0',# path to PCIe device or "sim"
        defaultFile         = 'config/defaults.yml',
        emuMode             = False,
        pollEn              = False, # Enable automatic polling registers
        initRead            = False, # Read all registers at start of the system
        promProg            = False, # Flag to disable all devices not related to PROM programming
        top_level           = f'{os.getcwd()}/', #used when calling the class from the jupiter notebook
        dataViewer          = False,  # Used to disable the viewer
        viewAsic            = 0,
        otherViewers        = False,  # Used to enable the scope viewer and the environmental viewer
        numOfAsics          = 4,
        numClusters         = 14,    # NUmber of enabled clusters in the pixel matrix
        numOfScopes         = 0,
        numOfAdcmons        = 0,
        timingMessage       = True,
        zmqSrvEn            = True,  # Flag to include the ZMQ server
        justCtrl            = False,
        **kwargs):

        self.sim            = (dev == 'sim')
        self.defaultFile    = defaultFile
        self.emuMode        = emuMode
        self.promProg       = promProg
        self.top_level      = top_level
        self.dataViewer     = dataViewer
        self.viewAsic       = viewAsic
        self.otherViewers   = otherViewers
        self.numOfAsics     = numOfAsics
        self.numClusters    = numClusters
        self.numOfScopes    = numOfScopes
        self.numOfAdcmons   = numOfAdcmons
        self.timingMessage  = timingMessage
        self.justCtrl       = justCtrl
        # Pass custom value to parent via super function
        super().__init__(**kwargs)

        #################################################################
        if zmqSrvEn:
            self.zmqServer = pyrogue.interfaces.ZmqServer(root=self, addr='127.0.0.1', port=0)
            self.addInterface(self.zmqServer)
        #################################################################

        if (self.sim):
            # Set the timeout
            self._timeout = 100000000 # firmware simulation slow and timeout base on real time (not simulation time)

        else:
            # Set the timeout
            self._timeout = 5.0 # 5.0 seconds default

        # File writer
        self.dataWriter = pr.utilities.fileio.StreamWriter()
        self.add(self.dataWriter)

        self.dataStream    = [None for i in range(self.numOfAsics)]

        if (self.otherViewers == True):
            self.oscopeStream  = [None for i in range(self.numOfScopes)]
            self.adcMonStream  = [None for i in range(self.numOfAdcmons)]

        if (self.dataViewer == True):
            # Create rateDrop, Unbatcher and filter if needed
            self.rate          = [rogue.interfaces.stream.RateDrop(True, 0.15) for lane in range(self.numOfAsics)]

        ##########################################################################################################
        # Check if not VCS simulation
        if (not self.sim):
            self._pollEn   = pollEn
            self._initRead = initRead
            if (self.justCtrl == False) :
            # # Map the DMA streams
                for lane in range(self.numOfAsics):
                    self.dataStream[lane] = rogue.hardware.axi.AxiStreamDma(dev, 0x100 * lane + 0, 1)

            self.srpStream = rogue.hardware.axi.AxiStreamDma(dev, 0x100 * 5 + 0, 1)

            self.ssiCmdStream = rogue.hardware.axi.AxiStreamDma(dev, 0x100 * 5 + 1, 1)

            # self.xvcStream = rogue.hardware.axi.AxiStreamDma(dev, 0x100 * 5 + 2, 1)

            if (self.otherViewers == True and self.justCtrl == False):
                for vc in range(self.numOfAdcmons):
                    self.adcMonStream[vc] = rogue.hardware.axi.AxiStreamDma(dev, 0x100 * 6 + vc, 1)
                for vc in range(self.numOfScopes):
                    self.oscopeStream[vc] = rogue.hardware.axi.AxiStreamDma(dev, 0x100 * 7 + vc, 1)

            # # # Create (Xilinx Virtual Cable) XVC on localhost
            # self.xvc = rogue.protocols.xilinx.Xvc(2542)
            # self.addProtocol(self.xvc)

            # # # Connect xvcStream to XVC module
            # self.xvcStream == self.xvc

            # # Create SRPv3
            self.srp = rogue.protocols.srp.SrpV3()

            # # Connect SRPv3 to srpStream
            pyrogue.streamConnectBiDir(self.srpStream,self.srp)


        else:
            # Start up flags are FALSE for simulation mode
            self._pollEn   = False
            self._initRead = False

            # Map the simulation DMA streams
            # 2 TCP ports per stream
            self.srp = rogue.interfaces.memory.TcpClient('localhost', 24000)

            if (self.justCtrl == False) : 
                for lane in range(self.numOfAsics):
                    # 2 TCP ports per stream
                    self.dataStream[lane] = rogue.interfaces.stream.TcpClient('localhost', 24002 + 2 * lane)

            if (self.otherViewers == True and self.justCtrl == False):
                for vc in range(self.numOfAdcmons):
                    self.adcMonStream[vc] = rogue.interfaces.stream.TcpClient('localhost', 24016 + 2 * vc)
                for vc in range(self.numOfScopes):
                    self.oscopeStream[vc] = rogue.interfaces.stream.TcpClient('localhost', 24026 + 2 * vc)

            # 2 TCP ports per stream
            self.ssiCmdStream = rogue.interfaces.stream.TcpClient('localhost', 24012)

            self._cmd = rogue.protocols.srp.Cmd()

            # # Connect ssiCmd to ssiCmdStream
            pyrogue.streamConnect(self._cmd, self.ssiCmdStream)

            ##########################################################################################################

        if (self.dataViewer == True and self.justCtrl == False):
            self.add(ePixUhr100kHz.DataReceiverEpixUHR100kHz(name = f"DataReceiver{viewAsic}", numClusters = self.numClusters, timingMessage = self.timingMessage))
            self.dataStream[viewAsic] >> self.rate[viewAsic] >> getattr(self, f"DataReceiver{viewAsic}")

            if (viewAsic ==0):
                @self.command()
                def DisplayViewer0():
                    subprocess.Popen(["python", self.top_level+"scripts/runLiveDisplay.py", "--dataReceiver", "rogue://0/root.DataReceiver0", "image", "--title", "ASIC 1"], shell=False)
            
            if (viewAsic ==1):
                @self.command()
                def DisplayViewer1():
                    subprocess.Popen(["python", self.top_level+"scripts/runLiveDisplay.py", "--dataReceiver", "rogue://0/root.DataReceiver1", "image", "--title", "ASIC 2"], shell=False)
            
            if (viewAsic ==2):
                @self.command()
                def DisplayViewer2():
                    subprocess.Popen(["python", self.top_level+"scripts/runLiveDisplay.py", "--dataReceiver", "rogue://0/root.DataReceiver2", "image", "--title", "ASIC 3"], shell=False)
            
            if (viewAsic ==3):
                @self.command()
                def DisplayViewer3():
                    subprocess.Popen(["python", self.top_level+"scripts/runLiveDisplay.py", "--dataReceiver", "rogue://0/root.DataReceiver3", "image", "--title", "ASIC 4"], shell=False)

        if (self.otherViewers == True and self.justCtrl == False):
            # for vc in range(self.numOfAdcmons):
            #     self.add(
            #         EnvDataReceiver(
            #             config = envConf[vc],
            #             clockT = 6.4e-9,
            #             rawToData = lambda raw: (2.5 * float(raw & 0xffffff)) / 16777216,
            #             name = f"EnvData{vc}"
            #         )
            #     )
            #     self.adcMonStream[vc] >> getattr(self, f"EnvData{vc}")

            for vc in range(self.numOfScopes):
                self.add(ScopeDataReceiver(name = f"ScopeData{vc}"))
                self.oscopeStream[vc] >> getattr(self, f"ScopeData{vc}")

            @self.command()
            def DisplayPseudoScope0():
                subprocess.Popen(["python", self.top_level+"scripts/runLiveDisplay.py", "--dataReceiver", "rogue://0/root.ScopeData0", "pseudoscope", "--title", "ScopeData0"], shell=False)

            @self.command()
            def DisplayEnvMonitor0():
                subprocess.Popen(["python", self.top_level+"scripts/runLiveDisplay.py", "--dataReceiver", "rogue://0/root.EnvMonitor0", "monitor", "--title", "EnvMonitor0"], shell=False)
            #################################################################

        # Add Devices
        self.add(leapCommon.Core(
            name         = 'Core',
            offset       = 0x0000_0000,
            memBase      = self.srp,
            sim          = self.sim,
            promProg     = self.promProg,
            expand       = False,
            pgpLaneVc    = [1],
            leapWriteEn  = True,
        ))

        self.add(epixUhrDev.App(
            name     = 'App',
            offset   = 0x8000_0000,
            memBase  = self.srp,
            sim      = self.sim,
            enabled  = not self.promProg,
            expand   = False,
        ))

        #################################################################
        # Connect dataStream to data writer
        if (self.justCtrl == False) :
            for lane in range(numOfAsics):
                self.dataStream[lane] >> self.dataWriter.getChannel(lane)

    def start(self,**kwargs):
        super(Root, self).start(**kwargs)

        # Check if not in simulation mode
        if self.sim is False:
            # Print the Current Firmware Version Information
            click.secho('###############################################################################', bg='green')
            self.Core.AxiVersion.printStatus()
            click.secho('###############################################################################', bg='green')

            pll = self.Core.Si5345Pll
            pll.enable.set(True)
            pll.LoadCsvFile('/cds/home/m/melchior/git/EVERYTHING_EPIX_UHR/epix-uhr-gtreadout-dev/software/config/pll/Si5345-B-156MHZ-out-0-5-and-7-v2-Registers.csv')

            self.App.TimingRx.UseMiniTpg.set(1)

            # TODO: Commented out because it does not work with hardware right now
            # if self.emuMode == 0:
            #     pll = self.Core.Si5345Pll
            #     pll.enable.set(True)
            #     # Load the PLL configurations
            #     pll.LoadCsvFile(self.top_level + 'config/pll/Si5345-B-156MHZ-out2-3-7.csv')

            #     # Wait for the PLL to lock
            #     click.secho('Waiting for PLL to lock ...', bg='blue')
            #     while(pll.Locked.get() == False):
            #         time.sleep(0.1)

            #     click.secho('PLL lock established', bg='blue')
