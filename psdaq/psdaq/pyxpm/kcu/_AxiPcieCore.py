#-----------------------------------------------------------------------------
# This file is part of the 'axi-pcie-core'. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the 'axi-pcie-core', including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import pyrogue              as pr
from psdaq.utils import enable_cameralink_gateway # to get surf
import surf.axi             as axi
import surf.devices.micron  as micron
import surf.devices.nxp     as nxp
import surf.xilinx          as xil

import xpm
import kcu
import click
import struct

class QSFPMonitor(pr.Device):
    def __init__(self,
                 name        = 'QSFPMonitor',
                 description = 'QSFP monitoring and diagnostics',
                 **kwargs):
        super().__init__(
            name        = name,
            description = description,
            **kwargs
        )

        self.add(pr.RemoteVariable(
            name      = 'page',
            offset    = (127<<2),
            bitSize   = 8,
            verify    = False,
            mode      = 'RW'
        ))

        self.add(pr.RemoteVariable(
            name      = 'TmpVccBlock',
            offset    = (22<<2),
            bitSize   = 32*6,
            mode      = 'RO'
        ))

        self.add(pr.RemoteVariable(
            name      = 'RxPwrBlock',
            offset    = (34<<2),
            bitSize   = 32*8,
            mode      = 'RO'
        ))

        self.add(pr.RemoteVariable(
            name      = 'TxBiasBlock',
            offset    = (42<<2),
            bitSize   = 32*8,
            mode      = 'RO'
        ))

        self.add(pr.RemoteVariable(
            name      = 'BaseIdBlock',
            offset    = (128<<2),
            bitSize   = 32*3,
            mode      = 'RO'
        ))

        self.add(pr.RemoteVariable(
            name      = 'DateBlock',
            offset    = (212<<2),
            bitSize   = 32*6,
            mode      = 'RO'
        ))

        self.add(pr.RemoteVariable(
            name      = 'DiagnType',
            offset    = (220<<2),
            bitSize   = 32,
            mode      = 'RO'
        ))

        self.add(pr.RemoteVariable(
            name      = 'VendorName',
            offset    = (148<<2),
            bitSize   = 32*16,
            mode      = 'RO'
        ))

    def getDate(self):
        self.page.set(0)
        v = self.DateBlock.get()
        def toChar(sh,w=v):
            return (w>>(32*sh))&0xff

        r = '{:c}{:c}/{:c}{:c}/20{:c}{:c}'.format(toChar(2),toChar(3),toChar(4),toChar(5),toChar(0),toChar(1))
        return r

    def getRxPwr(self):  #mW
        #self.page.set(0)
        v = self.RxPwrBlock.get()

        def word(a,o):
            return (a >> (32*o))&0xff
        def tou16(a,o):
            return struct.unpack('H',struct.pack('BB',word(a,o+1),word(a,o)))[0]
        def pwr(lane,v=v):
            p = tou16(v,2*lane)
            return p * 0.0001
                
        return (pwr(0),pwr(1),pwr(2),pwr(3))


    def getTxBiasI(self):  #mA
        #self.page.set(0)
        v = self.TxBiasBlock.get()

        def word(a,o):
            return (a >> (32*o))&0xff
        def tou16(a,o):
            return struct.unpack('H',struct.pack('BB',word(a,o+1),word(a,o)))[0]
        def pwr(lane,v=v):
            p = tou16(v,2*lane)
            return p * 0.002
                
        return (pwr(0),pwr(1),pwr(2),pwr(3))

class AxiPcieCore(pr.Device):
    """This class maps to axi-pcie-core/shared/rtl/AxiPcieReg.vhd"""
    def __init__(self,
                 name        = 'AxiPcieCore',
                 description = 'Base components of the PCIe firmware core',
                 useBpi      = False,
                 useSpi      = False,
                 numDmaLanes = 2,
                 boardType   = None,
                 **kwargs):
        super().__init__(description=description, **kwargs)

        self.numDmaLanes = numDmaLanes
        self.startArmed  = True

        # PCI PHY status
        self.add(xil.AxiPciePhy(
            offset       = 0x10000,
            expand       = False,
        ))

        # AxiVersion Module
        self.add(kcu.PcieAxiVersion(
            offset       = 0x20000,
            expand       = False,
        ))

        # Check if using BPI PROM
        if (useBpi):
            self.add(micron.AxiMicronP30(
                offset       =  0x30000,
                expand       =  False,
                hidden       =  True,
            ))

        # Check if using SPI PROM
        if (useSpi):
            for i in range(2):
                self.add(micron.AxiMicronN25Q(
                    name         = f'AxiMicronN25Q[{i}]',
                    offset       =  0x40000 + (i * 0x10000),
                    expand       =  False,
                    addrMode     =  True, # True = 32-bit Address Mode
                    hidden       =  True,
                ))

        # DMA AXI Stream Inbound Monitor
        self.add(axi.AxiStreamMonAxiL(
            name        = 'DmaIbAxisMon',
            offset      = 0x60000,
            numberLanes = self.numDmaLanes,
            expand      = False,
        ))

        # DMA AXI Stream Outbound Monitor
        self.add(axi.AxiStreamMonAxiL(
            name        = 'DmaObAxisMon',
            offset      = 0x68000,
            numberLanes = self.numDmaLanes,
            expand      = False,
        ))

        # Check for the SLAC GEN4 PGP Card
        if boardType == 'SlacPgpCardG4':

            pass

        elif boardType == 'Kcu1500':

            self.add(pr.RemoteVariable(
                name        = 'I2cMux',
                description = 'Bus selector',
                offset      = 0x7_0000,
                bitSize     = 8,
                bitOffset   = 0,
                base        = pr.UInt,
                mode        = 'RW',
                verify      = False
            ))

            self.add(QSFPMonitor(
                name   = 'QSFP',
                offset = 0x7_0400,
            ))

            self.add(xpm.Si570(
                name    = 'Si570',
                offset  = 0x7_0800,
                enabled = False
            ))


    def _start(self):
        super()._start()
        DMA_SIZE_G = self.AxiVersion.DMA_SIZE_G.get()
        if ( self.numDmaLanes is not DMA_SIZE_G ) and (self.startArmed):
            self.startArmed = False
            click.secho(f'WARNING: {self.path}.numDmaLanes = {self.numDmaLanes} != {self.path}.AxiVersion.DMA_SIZE_G = {DMA_SIZE_G}', bg='cyan')
