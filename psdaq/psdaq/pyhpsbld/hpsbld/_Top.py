#!/usr/bin/env python3
##############################################################################
## This file is part of 'HPSBLD'.
## It is subject to the license terms in the LICENSE.txt file found in the 
## top-level directory of this distribution and at: 
##    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
## No part of 'EPIX', including this file, 
## may be copied, modified, propagated, or distributed except according to 
## the terms contained in the LICENSE.txt file.
##############################################################################

import rogue
import rogue.hardware.axi

import pyrogue as pr
import pyrogue.protocols
import time

import psdaq.pyxpm.surf.axi                     as axi
import psdaq.pyxpm.surf.xilinx                  as xil
import psdaq.pyxpm.surf.devices.ti              as ti
import psdaq.pyxpm.surf.devices.micron          as micron
import psdaq.pyxpm.surf.devices.microchip       as microchip
import psdaq.pyxpm.LclsTimingCore               as timing
import psdaq.pyhpsbld.hpsbld        as hps

class Top(pr.Device):

    def __init__(   self,       
            name        = "Top",
            description = "Container for HPS BLD",
            ipAddr      = '10.0.1.101',
            memBase     = 0,
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)
        

        ################################################################################################################
        # UDP_SRV_XVC_IDX_C         => 2542,  -- Xilinx XVC 
        # UDP_SRV_SRPV0_IDX_C       => 8192,  -- Legacy SRPv0 register access (still used for remote FPGA reprogramming)
        # UDP_SRV_RSSI0_IDX_C       => 8193,  -- Legacy Non-interleaved RSSI for Register access and ASYNC messages
        # UDP_SRV_RSSI1_IDX_C       => 8194,  -- Legacy Non-interleaved RSSI for bulk data transfer
        # UDP_SRV_BP_MGS_IDX_C      => 8195,  -- Backplane Messaging
        # UDP_SRV_TIMING_IDX_C      => 8197,  -- Timing ASYNC Messaging
        # UDP_SRV_RSSI_ILEAVE_IDX_C => 8198);  -- Interleaved RSSI         
        ################################################################################################################

        # Create SRP/ASYNC_MSG interface
        if False:
            # UDP only
            self.udp = rogue.protocols.udp.Client(ipAddr,8192,0)
            
            # Connect the SRPv0 to RAW UDP
            self.srp = rogue.protocols.srp.SrpV0()
            pyrogue.streamConnectBiDir( self.srp, self.udp )

        if True:
            self.rudp = pyrogue.protocols.UdpRssiPack( name='rudpReg', host=ipAddr, port=8193, packVer = 1, jumbo = False)

            # Connect the SRPv3 to tDest = 0x0
            self.srp = rogue.protocols.srp.SrpV3()
            pr.streamConnectBiDir( self.srp, self.rudp.application(dest=0x0) )

            # Create stream interface
            #self.stream = pr.protocols.UdpRssiPack( name='rudpData', host=ipAddr, port=8194, packVer = 1, jumbo = False)

        ######################################################################
        
        # Add devices
        self.add(axi.AxiVersion( 
            memBase = self.srp,
            offset  = 0x00000000, 
            expand  = False,
        ))

        self.add(xil.AxiSysMonUltraScale(   
            memBase = self.srp,
            offset       =  0x01000000, 
            expand       =  False
        ))
        
        self.add(micron.AxiMicronN25Q(
            memBase = self.srp,
            name         = "MicronN25Q",
            offset       = 0x02000000,
            addrMode     = True,                                    
            expand       = False,                                    
            hidden       = True,                                    
        ))        

        self.add(microchip.AxiSy56040(    
            memBase = self.srp,
            offset       =  0x03000000, 
            expand       =  False,
            description  = "\n\
                Timing Crossbar:  https://confluence.slac.stanford.edu/x/m4H7D   \n\
                -----------------------------------------------------------------\n\
                OutputConfig[0] = 0x0: Connects RTM_TIMING_OUT0 to RTM_TIMING_IN0\n\
                OutputConfig[0] = 0x1: Connects RTM_TIMING_OUT0 to FPGA_TIMING_IN\n\
                OutputConfig[0] = 0x2: Connects RTM_TIMING_OUT0 to BP_TIMING_IN\n\
                OutputConfig[0] = 0x3: Connects RTM_TIMING_OUT0 to RTM_TIMING_IN1\n\
                -----------------------------------------------------------------\n\
                OutputConfig[1] = 0x0: Connects FPGA_TIMING_OUT to RTM_TIMING_IN0\n\
                OutputConfig[1] = 0x1: Connects FPGA_TIMING_OUT to FPGA_TIMING_IN\n\
                OutputConfig[1] = 0x2: Connects FPGA_TIMING_OUT to BP_TIMING_IN\n\
                OutputConfig[1] = 0x3: Connects FPGA_TIMING_OUT to RTM_TIMING_IN1 \n\
                -----------------------------------------------------------------\n\
                OutputConfig[2] = 0x0: Connects Backplane DIST0 to RTM_TIMING_IN0\n\
                OutputConfig[2] = 0x1: Connects Backplane DIST0 to FPGA_TIMING_IN\n\
                OutputConfig[2] = 0x2: Connects Backplane DIST0 to BP_TIMING_IN\n\
                OutputConfig[2] = 0x3: Connects Backplane DIST0 to RTM_TIMING_IN1\n\
                -----------------------------------------------------------------\n\
                OutputConfig[3] = 0x0: Connects Backplane DIST1 to RTM_TIMING_IN0\n\
                OutputConfig[3] = 0x1: Connects Backplane DIST1 to FPGA_TIMING_IN\n\
                OutputConfig[3] = 0x2: Connects Backplane DIST1 to BP_TIMING_IN\n\
                OutputConfig[3] = 0x3: Connects Backplane DIST1 to RTM_TIMING_IN1\n\
                -----------------------------------------------------------------\n"\
            ))
                            
        self.add(ti.AxiCdcm6208(     
            memBase = self.srp,
            offset       =  0x05000000, 
            expand       =  False,
        ))

        self.add(timing.TimingFrameRx(
            memBase = self.srp,
            name = 'UsTiming',
            offset = 0x08000000,
        ))

        self.add(hps.BldControl(
            memBase = self.srp,
            name   = 'BldControl',
            offset = 0x09000000,
        ))        

