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
import rogue
import click
import struct
import time

import kcu

rogue.Version.minVersion('5.1.0')
# rogue.Version.exactVersion('5.1.0')

#############################################
# Debug console printout
#############################################
class DmaHandle(rogue.interfaces.stream.Slave):

    def __init__(self, name):
        super().__init__()

        self.name = name
        self._rxErr = [0 for i in range(14)]
        self._rxRcv = [0 for i in range(14)]
        self._handle = None

    def _acceptFrame(self, frame):

        fidPeriod   = 1400e-6/1300.
    
        with frame.lock():
            size = frame.getPayload()

            msg = bytearray(size)
            frame.read(msg,0)
            
            if self._handle:
                self._handle(msg)
            else:
                offset = 4
                # LinkStatus
                linkS = []
                for i in range(14):
                    if i<8:
                        w = struct.unpack_from('<LLL',msg,offset)
                        u = (w[2]<<64) + (w[1]<<32) + w[0]
                        d = {'Lane':i,'txDone':((u>>0)&1),'txRdy ':((u>>1)&1),'rxDone':((u>>2)&1),'rxRdy ':((u>>3)&1)}

                        v = (u>>5)&0xffff
                        d['rxErr '] = v-self._rxErr[i]
                        self._rxErr[i] = v
                    
                        v = (u>>21)&0xffffffff
                        d['rxRcv '] = v-self._rxRcv[i]
                        self._rxRcv[i] = v

                        d['remId '] = (u>>54)&0xffffffff
                        print(d)
                    offset += 12

                # GroupStatus

                def bytes2Int(msg,offset):
                    b = struct.unpack_from('<BBBBB',msg,offset)
                    offset += 5
                    w = 0
                    for i,v in enumerate(b):
                        w += v<<(8*i)
                    return (w,offset)

                for i in range(8):
                    for k in range(32):
                        offset += 8
                    (l0Ena   ,offset) = bytes2Int(msg,offset)
                    (l0Inh   ,offset) = bytes2Int(msg,offset)
                    (numL0   ,offset) = bytes2Int(msg,offset)
                    (numL0Inh,offset) = bytes2Int(msg,offset)
                    (numL0Acc,offset) = bytes2Int(msg,offset)
                    offset += 1
                    rT = l0Ena*fidPeriod
                    d = {'Group':i,'L0Ena':l0Ena,'L0Inh':l0Inh,'NumL0':numL0,'rTim':rT}
                    print(d)

                for i in range(2):
                    offset += 1

                w = struct.unpack_from('<LLLL',msg,offset)
                offset += 16
                d = {'bpClk ':w[0]&0xfffffff,'fbClk ':w[1]&0xfffffff,'recClk':w[2]&0xfffffff,'phyClk':w[3]&0xfffffff}
                print(d)

                
class DevRoot(pr.Root):

    def __init__(self,
                 dev            = None,
                 pollEn         = True,  # Enable automatic polling registers
                 initRead       = True,  # Read all registers at start of the system
                 dataDebug      = False,
                 enableConfig   = False,
                 enVcMask       = 0x3, # Enable lane mask
                 **kwargs):

        print(f'DevRoot dataDebug {dataDebug}')

        # Set local variables
        self.dev            = dev
        self.enableConfig   = enableConfig
        self.defaultFile    = []

        # Check for simulation
        if dev == 'sim':
            kwargs['timeout'] = 100000000 # 100 s
        else:
            kwargs['timeout'] = 5000000 # 5 s

        # Pass custom value to parent via super function
        super().__init__(
            pollEn      = pollEn,
            initRead    = initRead,
            **kwargs)

        # Unhide the RemoteVariableDump command
        self.RemoteVariableDump.hidden = False

        # Create memory interface
        self.memMap = kcu.createAxiPcieMemMap(dev, 'localhost', 8000)
        self.memMap.setName('PCIe_Bar0')

        # Instantiate the top level Device and pass it the memory map
        self.add(kcu.DevPcie(
            name        = 'XPM',
            memBase     = self.memMap,
        ))

        # Create empty list
        self.dmaStreams     = [[None for x in range(4)] for y in range(4)]
        self._srp           = [None for x in range(4)]
        self._dbg           = [None for x in range(4)]
        self.enVcMask       = [False for x in range(4)]

        # Create DMA streams
        lane = 0
        if True:
            for vc in range(4):
                if enVcMask & (0x1 << vc):
                    self.enVcMask[vc] = True
                    if (dev is not 'sim'):
                        self.dmaStreams[lane][vc] = rogue.hardware.axi.AxiStreamDma(dev,(0x100*lane)+vc,1)
                    else:
                        self.dmaStreams[lane][vc] = rogue.interfaces.stream.TcpClient('localhost', (8000+2)+(512*lane)+2*vc)

                    self._dbg[vc] = DmaHandle(name='DmaHandle')
                    # Connect the streams
                    self.dmaStreams[lane][vc] >> self._dbg[vc]

        print(f'enVcMask {enVcMask} {self.enVcMask}')

        # Check if not doing simulation
        if (dev is not 'sim'):

            # Create the stream interface
            lane = 0
            if True:

                # Check if PGP[lane].VC[0] = SRPv3 (register access) is enabled
                if self.enVcMask[0]:
                    # Create the SRPv3
                    self._srp[lane] = rogue.protocols.srp.SrpV3()
                    self._srp[lane].setName(f'SRPv3[{lane}]')
                    # Connect DMA to SRPv3
                    self.dmaStreams[lane][0] = self._srp[lane]

        # Check if PGP[lane].VC[0] = SRPv3 (register access) is enabled
        if self.enVcMask[0]:
            self.add(pr.LocalVariable(
                name        = 'RunState',
                description = 'Run state status, which is controlled by the StopRun() and StartRun() commands',
                mode        = 'RO',
                value       = False,
            ))

    def handle(self, cb):
        for vc in range(4):
            self._dbg[vc]._handle = cb

    def start(self, **kwargs):
        super().start(**kwargs)

        self.XPM.start()

#        # Hide all the "enable" variables
#        for enableList in self.find(typ=pr.EnableVariable):
#            # Hide by default
#            enableList.hidden = True

        # Check if simulation
        if (self.dev is 'sim'):
            pass

        # Check if PGP[lane].VC[0] = SRPv3 (register access) is enabled
        elif self.enVcMask[0]:
            self.ReadAll()
            self.ReadAll()

            # Load the configurations
            if self.enableConfig:

                # Read all the variables
                self.ReadAll()
                self.ReadAll()

                # Load the YAML configurations
                defaultFile.extend(self.defaultFile)
                print(f'Loading {defaultFile} Configuration File...')
                self.LoadConfig(defaultFile)

    # Function calls after loading YAML configuration
    def initialize(self):
        super().initialize()
        self.CountReset()
