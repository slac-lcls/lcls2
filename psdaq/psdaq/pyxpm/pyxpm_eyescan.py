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

from psdaq.utils.enable_cameralink_gateway
import surf.axi                     as axi

import psdaq.pyxpm.xpm              as xpm

from p4p.client.thread import Context

class Top(pr.Root):

    def __init__(   self,       
                    name        = "Top",
                    description = "Container for XPM",
                    ipAddr      = 'localhost',
                    memBase     = 0,
                    **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        self.zmqServer = pr.interfaces.ZmqServer(root=self, addr='*', port=0)
        self.addInterface(self.zmqServer)

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

        LinksAMC0 = [ 
                     ['Link[0]',0x80100000],
                     ['Link[1]',0x80110000],
                     ['Link[2]',0x80120000],
                     ['Link[3]',0x80130000],
                     ['Link[4]',0x80140000],
                     ['Link[5]',0x80150000],
                     ['Link[6]',0x80160000],
                     ['Link[7]',0x80200000],
                     ['Link[8]',0x80210000],
                     ['Link[9]',0x80220000],
                     ['Link[10]',0x80230000],
                     ['Link[11]',0x80240000],
                     ['Link[12]',0x80250000],
                     ['Link[13]',0x80260000]  
        ]


        for i in range(len(LinksAMC0)):
            self.add(xpm.XpmGth(
                memBase = self.srp,
                name   = LinksAMC0[i][0],
                offset = LinksAMC0[i][1],
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
            linkMonitoring = True
        ))

def main():
    global prefix
    prefix = ''

    parser = argparse.ArgumentParser(prog=sys.argv[0], description='host PVs for XPM')

    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')
    parser.add_argument('--ip', type=str, required=True, help="IP address" )
    parser.add_argument('--link', type=int, required=False, help="Link identifier" )
    parser.add_argument('--hseq', type=int, required=False, help="High speed equalizer setting" )
    parser.add_argument('--eye', action='store_true', help='Generate eye diagram')
    parser.add_argument('--bathtub', action='store_true', help='Generate bathtub curve')
    parser.add_argument('--gui', action='store_true', help='Bring up GUI')
    parser.add_argument('--target', type=float, required=False, help="BET Target" )
    parser.add_argument('--forceLoopback', action='store_true', help='Set link in loopback mode')

    args = parser.parse_args()
    if args.verbose:
        setVerbose(True)

    #if args.eye == False and args.bathtub == False:
    #    print("Select at least one plot: --eye or --bathtub")
    #    return

    # Set base
    base = Top(
        name='XPM',
        description='',
        ipAddr = args.ip,
    )

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

    if args.link is None:
        linkid = 0
    else:
        linkid = args.link

    base.XpmApp.link.set(linkid)

    #GTH Config
    imageId = base.AxiVersion.ImageName.get()
    if imageId == 'xpm_noRTM' and linkid <= 6:
        linkid -= 1

    if linkid < 0 or linkid > 13:
        print("Error: linkid does not exists")
        base.stop()
        return
        
    loopback = base.XpmApp.loopback.get()
    if args.forceLoopback:
        print("Set in loopback")
        base.XpmApp.loopback.set(0x01)

        base.XpmApp.txReset.set(0x01)
        base.XpmApp.rxPllReset.set(0x01)
        base.XpmApp.rxReset.set(0x01)
        time.sleep(0.5)

        base.XpmApp.txReset.set(0x00)
        base.XpmApp.rxPllReset.set(0x00)
        base.XpmApp.rxReset.set(0x00)
        time.sleep(0.5)

    link_rxcdrlock = base.XpmApp.link_rxcdrlock.get()
    link_rxpmarstdone = base.XpmApp.link_rxpmarstdone.get()
    link_txpmarstdone = base.XpmApp.link_txResetDone.get()

    #base.XpmApp.link_eyescanrst.set(0x00)

    if link_rxcdrlock != 0x01 or link_rxpmarstdone != 0x01:
        print("Link not locked: CDR not locked, link not connected or data quality too bad")
        base.stop()
        return

    if args.hseq is not None:
        #High-speed repeater config
        device = base.HSRep[hsConfig[linkid]['dev']]
        device.CtrlEn.set(True)
        device.EQ_ch[hsConfig[linkid]['ch']].set(args.hseq)

    if args.target is None:
        target = 1e-8
    else:
        target = args.target

    link_rxRcvCnts = 0
    link_rxErrCnts = 0

    while link_rxRcvCnts < (1/target):
        base.XpmApp.link_gthCntRst.set(0x01)
        base.XpmApp.link_gthCntRst.set(0x00)
        time.sleep(0.1)
        link_rxRcvCnts += base.XpmApp.link_rxRcvCnts.get()*20
        link_rxErrCnts += base.XpmApp.link_rxErrCnts.get()

        print('BER: {:.2e} ({} err/ {} rcv)'.format((link_rxErrCnts/link_rxRcvCnts), link_rxErrCnts, link_rxRcvCnts))

    if args.gui:
        pyrogue.pydm.runPyDM(
            serverList  = base.zmqServer.address,
            sizeX       = 800,
            sizeY       = 800,
        )
  
    if args.bathtub:
        base.Link[linkid].bathtubPlot(target)

    if args.eye:
        base.Link[linkid].eyePlot(target=target)

    base.XpmApp.loopback.set(loopback)
    base.stop()

if __name__ == '__main__':
    main()
