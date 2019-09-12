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
import os
baseDir = os.path.dirname(os.path.realpath(__file__))
pr.addLibraryPath(baseDir)
pr.addLibraryPath(baseDir + '/../../surf/python')
pr.addLibraryPath(baseDir + '/../../axi-pcie-core/python')
pr.addLibraryPath(baseDir + '/../../lcls-timing-core/python')
pr.addLibraryPath(baseDir + '/../../lcls2-pgp-fw-lib/python')

import sys
import argparse
import time

import rogue.hardware.axi
import rogue.protocols

import XilinxKcu1500Pgp as kcu1500
import ClinkFeb         as feb

#################################################################

# Set the argument parser
parser = argparse.ArgumentParser()

# Convert str to bool
argBool = lambda s: s.lower() in ['true', 't', 'yes', '1']

# Add arguments
parser.add_argument(
    "--dev", 
    type     = str,
    required = False,
    default  = '/dev/datadev_0',
    help     = "path to device",
)  

parser.add_argument(
    "--version3", 
    type     = argBool,
    required = False,
    default  = False,
    help     = "true = PGPv3, false = PGP2b",
) 

parser.add_argument(
    "--mcs", 
    type     = str,
    required = True,
    help     = "path to mcs file",
)

parser.add_argument(
    "--lane", 
    type     = int,
    required = True,
    help     = "PGP lane index (range from 0 to 3)",
)  

# Get the arguments
args = parser.parse_args()

#################################################################

if ('_primary.mcs' in args.mcs) or ('_secondary.mcs' in args.mcs):
    raise ValueError(f'ERROR: --mcs looks like a PCIe image file (not FEB)' ) 

#################################################################

class MyRoot(kcu1500.Core):

    def __init__(self,
            name        = 'ClinkDev',
            description = 'Container for CameraLink Dev',
            dev         = '/dev/datadev_0',# path to PCIe device
            version3    = False,           # true = PGPv3, false = PGP2b
            pollEn      = False,           # Enable automatic polling registers
            initRead    = True,            # Read all registers at start of the system
            numLane     = 1,               # Number of PGP lanes
            **kwargs
        ):
        super().__init__(
            name        = name, 
            description = description, 
            dev         = dev, 
            version3    = version3, 
            pollEn      = pollEn, 
            initRead    = initRead, 
            numLane     = numLane, 
            **kwargs
        )
        
        # Check if not doing simulation
        if (dev != 'sim'):            
            
            # Create arrays to be filled
            self._srp = [None for lane in range(numLane)]
            
            # Create the stream interface
            for lane in range(numLane):
                    
                # SRP
                self._srp[lane] = rogue.protocols.srp.SrpV3()
                pr.streamConnectBiDir(self._dma[lane][0],self._srp[lane])
                         
                # CameraLink Feb Board
                self.add(feb.ClinkFeb(      
                    name        = (f'ClinkFeb[{lane}]'), 
                    memBase     = self._srp[lane], 
                    version3    = version3,
                    enableDeps  = [self.Hardware.PgpMon[lane].RxRemLinkReady], # Only allow access if the PGP link is established
                    # expand      = False,
                ))        
        
        # Start the system
        self.start(
            pollEn   = self._pollEn,
            initRead = self._initRead,
            timeout  = self._timeout,
        )

#################################################################

cl = MyRoot(
    dev      = args.dev,
    version3 = args.version3,
)
    
# Create useful pointers
AxiVersion = cl.ClinkFeb[args.lane].AxiVersion
PROM       = cl.ClinkFeb[args.lane].CypressS25Fl

# Read all the variables
cl.ReadAll()

if (cl.Hardware.PgpMon[args.lane].RxRemLinkReady.get()):
    print ( '###################################################')
    print ( '#                 Old Firmware                    #')
    print ( '###################################################')
    AxiVersion.printStatus()
else:
    # PGP Link down
    raise ValueError(f'Pgp[lane={args.lane}] is down')

# Program the FPGA's PROM
PROM.LoadMcsFile(args.mcs)

if(PROM._progDone):
    print('\nReloading FPGA firmware from PROM ....')
    AxiVersion.FpgaReload()
    time.sleep(5)
    print('\nReloading FPGA done')

    print ( '###################################################')
    print ( '#                 New Firmware                    #')
    print ( '###################################################')
    AxiVersion.printStatus()
else:
    print('Failed to program FPGA')

cl.stop()
exit()
