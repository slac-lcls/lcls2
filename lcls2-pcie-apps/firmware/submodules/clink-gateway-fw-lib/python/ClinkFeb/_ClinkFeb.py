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

import surf.axi             as axi
import surf.xilinx          as xil
import surf.devices.cypress as prom
import surf.devices.linear  as linear
import surf.devices.nxp     as nxp
import surf.protocols.clink as cl
import surf.protocols.pgp   as pgp
import ClinkFeb             as feb
        
class ClinkFeb(pr.Device):
    def __init__(   self,       
            name        = "ClinkFeb",
            description = "ClinkFeb Container",
            serial      = [None,None],
            camType     = [None,None],
            version3    = False, # true = PGPv3, false = PGP2b
            **kwargs):
        super().__init__(name=name, description=description, **kwargs) 

        # Add devices
        self.add(axi.AxiVersion( 
            name        = 'AxiVersion', 
            offset      = 0x00000000, 
            expand      = False,
        ))
        
        self.add(prom.CypressS25Fl(
            name        = 'CypressS25Fl', 
            offset      = 0x00001000, 
            hidden      = True, # Hidden in GUI because indented for scripting
        ))
        
        self.add(nxp.Sa56004x(      
            name        = 'BoardTemp', 
            description = 'This device monitors the board temperature and FPGA junction temperature', 
            offset      = 0x00002000, 
            expand      = False,
        ))
        
        self.add(linear.Ltc4151(
            name        = 'BoardPwr', 
            description = 'This device monitors the board power, input voltage and input current', 
            offset      = 0x00002400, 
            senseRes    = 20.E-3, # Units of Ohms
            expand      = False,
        ))
        
        self.add(xil.Xadc(
            name        = 'Xadc', 
            offset      = 0x00003000, 
            expand      = False,
        ))        
        
        self.add(cl.ClinkTop(
            offset      = 0x00100000,
            serial      = serial,
            camType     = camType,
            expand      = False,
        ))    

        self.add(feb.ClinkTrigCtrl(      
            name        = 'TrigCtrl[0]', 
            description = 'Channel A trigger control', 
            offset      = 0x00200000, 
            expand      = False,
        )) 

        self.add(feb.ClinkTrigCtrl(      
            name        = 'TrigCtrl[1]', 
            description = 'Channel B trigger control', 
            offset      = 0x00200100, 
            expand      = False,
        ))     

        for i in range(2):
        
            if (version3):
                self.add(pgp.Pgp3AxiL(            
                    name    = (f'PgpMon[{i}]'), 
                    offset  = (0x00400000 + i*0x2000), 
                    numVc   = 4,
                    writeEn = False,
                    expand  = False,
                )) 
                
                self.add(axi.AxiStreamMonitoring(            
                    name        = (f'PgpTxAxisMon[{i}]'), 
                    offset      = (0x00400000 + i*0x4000 + 0x4000), 
                    numberLanes = 4,
                    expand      = False,
                ))        

                self.add(axi.AxiStreamMonitoring(            
                    name        = (f'PgpRxAxisMon[{i}]'), 
                    offset      = (0x00400000 + i*0x4000 + 0x6000), 
                    numberLanes = 4,
                    expand      = False,
                ))
                
            else:
                self.add(pgp.Pgp2bAxi(            
                    name    = (f'PgpMon[{i}]'), 
                    offset  = (0x00400000 + i*0x6000 + 0*0x2000), 
                    writeEn = False,
                    expand  = False,
                ))           
     
                self.add(axi.AxiStreamMonitoring(            
                    name        = (f'PgpTxAxisMon[{i}]'), 
                    offset      = (0x00400000 + i*0x6000 + 1*0x2000), 
                    numberLanes = 4,
                    expand      = False,
                ))        

                self.add(axi.AxiStreamMonitoring(            
                    name        = (f'PgpRxAxisMon[{i}]'), 
                    offset      = (0x00400000 + i*0x6000 + 2*0x2000), 
                    numberLanes = 4,
                    expand      = False,
                ))                
                