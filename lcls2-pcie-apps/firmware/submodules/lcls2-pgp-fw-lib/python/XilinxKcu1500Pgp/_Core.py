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

import rogue.hardware.axi
import rogue.protocols
import pyrogue.interfaces.simulation

import XilinxKcu1500Pgp       as kcu1500
import surf.protocols.batcher as batcher
import LclsTimingCore         as timingCore

import time

class Core(pr.Root):

    def __init__(self,
            name        = 'Core',
            description = 'Container for XilinxKcu1500Pgp Core',
            dev         = '/dev/datadev_0',# path to PCIe device
            version3    = False,           # true = PGPv3, false = PGP2b
            pollEn      = True,            # Enable automatic polling registers
            initRead    = True,            # Read all registers at start of the system            
            numLane     = 4,               # Number of PGP lanes
            enVcMask    = 0xD,             # Enable lane mask: Don't connect data stream (VC1) by default because intended for C++ process
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)
        
        # Simplify the Command Tab
        self.WriteAll.hidden      = True        
        self.ReadAll.hidden       = True        
        self.SaveState.hidden     = True        
        self.SaveConfig.hidden    = True        
        self.LoadConfig.hidden    = True        
        self.Initialize.hidden    = True        
        self.SetYamlConfig.hidden = True        
        self.GetYamlConfig.hidden = True        
        self.GetYamlState.hidden  = True        
        self.HardReset.hidden     = True        
        self.CountReset.hidden    = True        
        self.ClearLog.hidden      = True        
        self.numLane              = numLane        
        
        # Enable Init after config
        self.InitAfterConfig._default = True
          
        # Create PCIE memory mapped interface
        if (dev != 'sim'):
            # BAR0 access
            self._memMap = rogue.hardware.axi.AxiMemMap(dev)     
            # Set the timeout
            self._timeout = 1.0 # 1.0 default
            # Start up flags
            self._pollEn   = pollEn
            self._initRead = initRead
        else:
            # FW/SW co-simulation
            self._memMap = rogue.interfaces.memory.TcpClient('localhost',8000)            
            # Set the timeout
            self._timeout = 100.0 # firmware simulation slow and timeout base on real time (not simulation time)
            # Start up flags
            self._pollEn   = False
            self._initRead = False
            
        # PGP Hardware on PCIe 
        self.add(kcu1500.Hardware(            
            memBase  = self._memMap,
            numLane  = numLane,
            version3 = version3,
            expand   = False,
        ))   

        # Create arrays to be filled
        self._dma = [[None for vc in range(4)] for lane in range(numLane)] # self._dma[lane][vc]
        if (dev == 'sim'):
            trigIndex     = 32 if version3 else 8
            self._pgp     = [[None for vc in range(4)] for lane in range(numLane)] # self._dma[lane][vc]
            self._pgpTrig = [None for lane in range(numLane)] # self._febTrig[lane]
            
        # Create the stream interface
        for lane in range(numLane):
        
            # Map the virtual channels 
            if (dev != 'sim'):
                # Loop through the Virtual channels
                for vc in range(4):
                    # Check the VC enable mask
                    if ((enVcMask>>vc) & 0x1):
                        # PCIe DMA Interface
                        self._dma[lane][vc] = rogue.hardware.axi.AxiStreamDma(dev,(0x100*lane)+vc,True)
            else:
                # PCIe DMA Interface
                self._dma[lane][0] = rogue.interfaces.stream.TcpClient('localhost',8002+(512*lane)+2*0) # VC0
                self._dma[lane][1] = rogue.interfaces.stream.TcpClient('localhost',8002+(512*lane)+2*1) # VC1
                self._dma[lane][2] = rogue.interfaces.stream.TcpClient('localhost',8002+(512*lane)+2*2) # VC2
                self._dma[lane][3] = rogue.interfaces.stream.TcpClient('localhost',8002+(512*lane)+2*3) # VC3
                # FEB Board Interface
                self._pgp[lane][0]  = rogue.interfaces.stream.TcpClient('localhost',7000+(34*lane)+2*0) # VC0
                self._pgp[lane][1]  = rogue.interfaces.stream.TcpClient('localhost',7000+(34*lane)+2*1) # VC1
                self._pgp[lane][2]  = rogue.interfaces.stream.TcpClient('localhost',7000+(34*lane)+2*2) # VC2
                self._pgp[lane][3]  = rogue.interfaces.stream.TcpClient('localhost',7000+(34*lane)+2*3) # VC3    
                self._pgpTrig[lane] = rogue.interfaces.stream.TcpClient('localhost',7000+(34*lane)+trigIndex) # OP-Code    
