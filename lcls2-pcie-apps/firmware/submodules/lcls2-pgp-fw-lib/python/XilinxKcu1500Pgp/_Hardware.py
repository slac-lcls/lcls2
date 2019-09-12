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

import axipcie as pcie
import surf.protocols.pgp as pgp
import surf.axi           as axi

import XilinxKcu1500Pgp

class Hardware(pr.Device):
    def __init__(   self,       
            name        = "Hardware",
            description = "Container for PCIe Hardware Registers",
            numLane     = 4,
            version3    = False,
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)
        
        # Add axi-pcie-core 
        self.add(pcie.AxiPcieCore(            
            offset      = 0x00000000,
            numDmaLanes = numLane,
            expand      = False,
        ))  

        # Add PGP Core 
        for i in range(numLane):
        
            if (version3):
                self.add(pgp.Pgp3AxiL(            
                    name    = (f'PgpMon[{i}]'), 
                    offset  = (0x00800000 + i*0x00010000 + 0*0x2000), 
                    numVc   = 4,
                    writeEn = True,
                    expand  = False,
                ))
                
            else:
                self.add(pgp.Pgp2bAxi(            
                    name    = (f'PgpMon[{i}]'), 
                    offset  = (0x00800000 + i*0x00010000), 
                    writeEn = True,
                    expand  = False,
                ))
        
            self.add(axi.AxiStreamMonitoring(            
                name        = (f'PgpTxAxisMon[{i}]'), 
                offset      = (0x00800000 + i*0x00010000 + 1*0x2000), 
                numberLanes = 4,
                expand      = False,
            ))        

            self.add(axi.AxiStreamMonitoring(            
                name        = (f'PgpRxAxisMon[{i}]'), 
                offset      = (0x00800000 + i*0x00010000 + 2*0x2000), 
                numberLanes = 4,
                expand      = False,
            ))           
            
        # Add Timing Core
        self.add(XilinxKcu1500Pgp.Timing(
            offset  = 0x00900000,
            numLane = numLane,
            expand  = False,
        ))
        