#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# This file is part of the 'Camera link gateway'. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the 'Camera link gateway', including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
import pyrogue as pr
import pyrogue.interfaces.simulation

from psadaq.utils import submod_cameralink_gateway  # to get surf

import rogue
import axipcie
import dev

import sys
import argparse
import logging
import time
import socket

class EyeScanRoot(pr.Root):

    def __init__(self,
                 datadev     = '/dev/datadev_0',# path to PCIe device
                 pollEn      = True,            # Enable automatic polling registers
                 initRead    = True,            # Read all registers at start of the system
                 numLanes    = 1,
                 defaultFile = None,
                 **kwargs):

        # Set local variables
        super().__init__(**kwargs)
        
        self.zmqServer = pr.interfaces.ZmqServer(root=self, addr='*', port=0)
        self.addInterface(self.zmqServer)

        kwargs['timeout'] = 5000000 # 5 s
          
        # Create PCIE memory mapped interface
        if (datadev != 'sim'):
            # Set the timeout
            self._timeout = 1.0 # 1.0 default
            # Start up flags
            self._pollEn   = pollEn
            self._initRead = initRead
        else:
            # Set the timeout
            self._timeout = 100.0 # firmware simulation slow and timeout base on real time (not simulation time)
            # Start up flags
            self._pollEn   = False
            self._initRead = False 

        self.numLanes    = numLanes  

        # Create memory interface
        self.memMap = axipcie.createAxiPcieMemMap(datadev, 'localhost', 8000)

        self.add(axipcie.AxiPcieCore(
            memBase     = self.memMap,
            offset      = 0x0000_0000
        ))

        self.add(dev.TimingRx(
            name        = 'timrx',
            memBase     = self.memMap,
            offset      = 0x00140000,
        ))

        # Instantiate the top level Device and pass it the memory map
        self.add(dev.EyeGth(
            name        = 'timing',
            memBase     = self.memMap,
            offset      = 0x00114000,
        ))

        pgplanes = [
            0x00901000,
            0x00911000,
            0x00921000,
            0x00931000,
            0x00941000,
            0x00951000,
            0x00961000,
            0x00971000,
        ]

        for i in range(len(pgplanes)):
            self.add(dev.EyeGth(
                name        = 'link[{}]'.format(i),
                memBase     = self.memMap,
                offset      = pgplanes[i],
            ))

    def start(self, **kwargs):
        super().start(**kwargs)

def main():
    global prefix
    prefix = ''

    parser = argparse.ArgumentParser(prog=sys.argv[0], description='Eyediag for HSD')

    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')
    parser.add_argument('--link', type=int, default=None, help="Link id ([-1 for timing link], [0 - 7 for pgp link])" )
    parser.add_argument('--dev', type=str, default="/dev/datadev_0", help="Device file (default: /dev/datadev_0)" )
    parser.add_argument('--eye', action='store_true', help='Generate eye diagram')
    parser.add_argument('--bathtub', action='store_true', help='Generate bathtub curve')
    parser.add_argument('--write', default=None, help='Write results to OPATH', metavar='OPATH')
    parser.add_argument('--gui', action='store_true', help='Bring up GUI')
    parser.add_argument('--target', type=float, required=False, help="BER Target" )

    args = parser.parse_args()
    if args.verbose:
        setVerbose(True)

    if args.dev is None:
        datadev = '/dev/datadev_0'
    else:
        datadev = args.dev

    ######################
    # Setup the system
    ######################
    root = EyeScanRoot(datadev)

    ###################### 
    # Start the system
    ######################
    root.start()

    if (args.link is None or args.link==-1) and root.timing.ES_EYE_SCAN_EN.get()==0:
        #  This requires an Rx PMA reset
        root.timing.ES_EYE_SCAN_EN.set(0x01)
        root.timing.ES_ERRDET_EN.set(0x01)
        
        root.timrx.RxReset.set(1)
        time.sleep(0.001)
        root.timrx.RxReset.set(0)

    if (args.link is None or args.link>=0):
        #  Don't know how to reset RXPMA for PGP links
        #  and they don't seem to need it
        pass

    if args.target is None:
        target = 1e-8
    else:
        target = args.target

    if args.gui:
        pyrogue.pydm.runPyDM(
            serverList  = root.zmqServer.address,
            sizeX       = 800,
            sizeY       = 800,
        )
  
    if args.bathtub:
        links = [i for i in range(0,8)] if args.link is None else [args.link]
        for link in links:
            if args.write:
                base = f'{args.write}/{socket.gethostname()}.{args.dev.split("/")[-1]}.{link}'
                fname = base+'.png'
                if link == -1:
                    result = root.timing.bathtubPlot(fname)
                else:
                    result = root.link[link].bathtubPlot(fname)
                f=open(base+'.dat',mode='w')
                f.write(f'BER:{result}')
                f.close()

            else:
                if link == -1:
                    result = root.timing.bathtubPlot()
                else:
                    result = root.link[link].bathtubPlot()

    if args.eye:
        if args.link == -1:
            root.timing.eyePlot(target=target)
        else: 
            root.link[args.link].eyePlot(target=target)


    root.stop()

if __name__ == '__main__':
    main()
        
