#!/usr/bin/env python

import sys
import pyrogue as pr
import argparse
import logging

import rogue
import rogue.hardware.axi

import pyrogue.protocols
import time
from psdaq.utils import submod_cameralink_gateway  # to get surf
import surf.axi                     as axi
from surf.devices.transceivers import Sfp

import psdaq.pyxpm.xpm              as xpm

class Top(pr.Device):

    def __init__(   self,       
                    name        = "Top",
                    description = "Container for XPM",
                    ipAddr      = '10.0.1.101',
                    memBase     = 0,
                    fidPrescale = 200,
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
            self.stream = pr.protocols.UdpRssiPack( name='rudpData', host=ipAddr, port=8194, packVer = 1, jumbo = False)

        ######################################################################
        
        # Add devices
        self.add(axi.AxiVersion( 
            memBase = self.srp,
            offset  = 0x00000000, 
            expand  = False,
        ))

        for i in range(2):
            for j in range(7):
                self.add( Sfp(
                    memBase = self.srp,
                    name    = f'Sfp{i}{j}',
                    offset  = 0x09000000+(i+1)*0x100000+j*0x1000,
                ))
        
def main():
    global prefix
    prefix = ''

    parser = argparse.ArgumentParser(prog=sys.argv[0], description='host PVs for XPM')

    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')
    parser.add_argument('--ip', type=str, required=True, help="IP address" )

    args = parser.parse_args()
    if args.verbose:
#        logging.basicConfig(level=logging.DEBUG)
        setVerbose(True)

    # Set base
    base = pr.Root(name='AMCc',description='') 

    base.add(Top(
        name   = 'XPM',
        ipAddr = args.ip,
    ))
    
    # Start the system
    base.start(
    )

    xpm = base.XPM
    axiv = base.XPM.AxiVersion

    # Print the AxiVersion Summary
    axiv.printStatus()

    cycle = 0
    while True:
        for amc in range(2):
            for i in range(7):
                sfp = getattr(base.XPM,f'Sfp{amc}{i}')
                print(f'-- Sfp{amc}{i}:')
                try:
                    print(f'vendor: {sfp.VendorNameRaw[0].get()}')
                except:
                    pass
                time.sleep(1)

if __name__ == '__main__':
    main()
