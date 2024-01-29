#!/usr/bin/env python

import sys
import time
import pyrogue as pr
import argparse
import logging

import rogue
import rogue.hardware.axi

import pyrogue.protocols
import pyrogue.pydm
import time

import cameralink_gateway  # to get surf
import surf.axi                     as axi

import psdaq.pyxpm.xpm              as xpm

from p4p.client.thread import Context

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
        if True:
            # UDP only
            self.udp = rogue.protocols.udp.Client(ipAddr,8192,0)
            
            # Connect the SRPv0 to RAW UDP
            self.srp = rogue.protocols.srp.SrpV0()
            pyrogue.streamConnectBiDir( self.srp, self.udp )

        if False:
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

        
        

        hsrParms = [ ['HSRep[0]',0x09000000],
                     ['HSRep[1]',0x09010000],
                     ['HSRep[2]',0x09020000],
                     ['HSRep[3]',0x09030000],
                     ['HSRep[4]',0x09040000],
                     ['HSRep[5]',0x09050000] ]
        for i in range(len(hsrParms)):
            self.add(xpm.Ds125br401(
                memBase = self.srp,
                name   = hsrParms[i][0],
                offset = hsrParms[i][1],
            ))

        self.add(xpm.XpmApp(
            memBase = self.srp,
            name   = 'XpmApp',
            offset = 0x80000000,
            fidPrescale = fidPrescale,
        ))

def eqScan(XPMID, base, hsConfig, link):

    settings = [
        0x00, 
        0x01, 
        0x02, 
        0x03, 
        0x07, 
        0x15, 
        0x0b, 
        0x0f, 
        0x55, 
        0x1f, 
        0x2f, 
        0x3f, 
        0xaa, 
        0x7f, 
        0xbf, 
        0xff, 
    ]

    device = base.XPM.HSRep[hsConfig[link]['dev']]
    device.CtrlEn.set(True)

    ctxt = Context('pva')

    ret = []

    print('Scan equalizer settings for link {} / XPM {}'.format(link, XPMID))
    for setting in settings:
        device.EQ_ch[hsConfig[link]['ch']].set(setting)
        time.sleep(2)

        linkRxReady = ctxt.get('DAQ:NEH:XPM:{}:LinkRxReady{}'.format(XPMID, link)).raw.value
        linkRxRcv = ctxt.get('DAQ:NEH:XPM:{}:LinkRxRcv{}'.format(XPMID, link)).raw.value
        linkRxErr = ctxt.get('DAQ:NEH:XPM:{}:LinkRxErr{}'.format(XPMID, link)).raw.value
        print('    Link[{}] status (eq={:02X}): {} (Rec: {} - Err: {})'.format(
            link, setting, 'Ready' if linkRxReady == 1 else 'Not ready', linkRxRcv, linkRxErr))

        if linkRxErr == 0 and linkRxReady == 1:
            ret.append(setting)

    if len(ret) > 0:
        v = ret[0]
        
    else:
        v = 0x2f

    device.EQ_ch[hsConfig[link]['ch']].set(v)
    device.CtrlEn.set(True)

    print('    [Configured] Set eq = 0x{:02X}'.format(v))

    return ret

def main():
    global prefix
    prefix = ''

    parser = argparse.ArgumentParser(prog=sys.argv[0], description='host PVs for XPM')

    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')
    parser.add_argument('--ip', type=str, required=True, help="IP address" )
    parser.add_argument('--xpmid', type=int, required=True, help="XPM Identifier" )
    parser.add_argument('--link', type=int, required=True, help="XPM Identifier" )

    args = parser.parse_args()
    if args.verbose:
        setVerbose(True)

    # Set base
    base = pr.Root(name='AMCc',description='') 

    top = Top(
        name   = 'XPM',
        ipAddr = args.ip,
    )
    base.add(top)
    
    # Start the system
    base.start()

    hsConfig = [
            {'dev': 0, 'ch': 3}, #AMC0 - Link 0 (link 0)
            {'dev': 0, 'ch': 2}, #AMC0 - Link 1 (link 1)
            {'dev': 0, 'ch': 1}, #AMC0 - Link 2 (link 2)
            {'dev': 0, 'ch': 0}, #AMC0 - Link 3 (link 3)
            {'dev': 1, 'ch': 3}, #AMC0 - Link 4 (link 4)
            {'dev': 1, 'ch': 2}, #AMC0 - Link 5 (link 5)
            {'dev': 1, 'ch': 1}, #AMC0 - Link 6 (link 6)
            {'dev': 3, 'ch': 3}, #AMC1 - Link 0 (link 7)
            {'dev': 3, 'ch': 2}, #AMC1 - Link 1 (link 8)
            {'dev': 3, 'ch': 1}, #AMC1 - Link 2 (link 9)
            {'dev': 3, 'ch': 0}, #AMC1 - Link 3 (link 10)
            {'dev': 4, 'ch': 3}, #AMC1 - Link 4 (link 11)
            {'dev': 4, 'ch': 2}, #AMC1 - Link 5 (link 12)
            {'dev': 4, 'ch': 1}, #AMC1 - Link 6 (link 13)
        ]

    r = eqScan(args.xpmid, base, hsConfig, args.link)
    #print(r)
    

if __name__ == '__main__':
    main()
