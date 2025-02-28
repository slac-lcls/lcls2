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

import cameralink_gateway  # to get surf

import rogue
import axipcie
import dev

import sys
import argparse
import logging
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

        #self.add(axipcie.AxiPcieCore(
        #    memBase     = self.memMap,
        #    offset      = 0x0000_0000
        #))

        pgplanes = [
            0x00A09000,
            0x00A19000,
            0x00A29000,
            0x00A39000
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
    parser.add_argument('--link', type=int, default=None, help="Link id (default is all)" )
    parser.add_argument('--dev', default='/dev/datadev_0', help="Device file (default: /dev/datadev_0)" )
    parser.add_argument('--eye', action='store_true', help='Generate eye diagram')
    parser.add_argument('--bathtub', action='store_true', help='Generate bathtub curve')
    parser.add_argument('--write', default=None, help='Write results to OPATH', metavar='OPATH')
    parser.add_argument('--gui', action='store_true', help='Bring up GUI')
    parser.add_argument('--target', type=float, required=False, help="BET Target" )

    args = parser.parse_args()
    if args.verbose:
        setVerbose(True)

    ######################
    # Setup the system
    ######################
    root = EyeScanRoot(args.dev)

    ###################### 
    # Start the system
    ######################
    root.start()

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
        links = [i for i in range(4)] if args.link is None else [args.link]
        for link in links:
            if args.write:
                base = f'{args.write}/{socket.gethostname()}.{args.dev.split("/")[-1]}.{link}'
                fname = base+'.png'
                result = root.link[link].bathtubPlot(fname)
                f=open(base+'.dat',mode='w')
                f.write(f'BER:{result}')
                f.close()
            else:
                result = root.link[link].bathtubPlot()

    if args.eye:
        root.link[args.link].eyePlot(target=target)


    root.stop()

if __name__ == '__main__':
    main()
        
