#!/usr/bin/env python3
##############################################################################
## This file is part of 'EPIX'.
## It is subject to the license terms in the LICENSE.txt file found in the 
## top-level directory of this distribution and at: 
##    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
## No part of 'EPIX', including this file, 
## may be copied, modified, propagated, or distributed except according to 
## the terms contained in the LICENSE.txt file.
##############################################################################

import time
import struct

import rogue
import rogue.hardware.axi

import pyrogue as pr
import surf.axi                     as axi

class TDetSemi(pr.Device):
    def __init__(self,
                 name        = 'TDetSemi',
                 description = 'Fake camera',
                 **kwargs):
        super().__init__(
            name        = name,
            description = description,
            **kwargs
        )

        self.add(pr.RemoteVariable(
            name      = 'rttBlock',
            offset    = 0x50,
            bitSize   = 32*4,
            mode      = 'RO'
        ))

    def getRTT(self):
        v = self.rttBlock.get()

        def fullToTrig(lane,v=v):
            return (v>>(32*lane))&0xffff
        def nfullToTrig(lane,v=v):
            return (v>>(32*lane+16))&0xfff

        return ( (fullToTrig(0),nfullToTrig(0)),
                 (fullToTrig(1),nfullToTrig(1)),
                 (fullToTrig(2),nfullToTrig(2)),
                 (fullToTrig(3),nfullToTrig(3)) )

class TDetTiming(pr.Device):
    def __init__(self,
                 name        = 'TDetTiming',
                 description = 'Template timed detector',
                 **kwargs):
        super().__init__(
            name        = name,
            description = description,
            **kwargs
        )

        self.add(pr.RemoteVariable(
            name      = 'rxRefClk',
            offset    = 0x10,
            bitSize   = 32,
            mode      = 'RO'
        ))

        self.add(pr.RemoteVariable(
            name      = 'txRefClk',
            offset    = 0x28,
            bitSize   = 32,
            mode      = 'RO'
        ))

    def getClkRates(self):
        rxp = self.rxRefClk.get()
        txp = self.txRefClk.get()
        time.sleep(1)
        rxn = self.rxRefClk.get()
        txn = self.txRefClk.get()
        return ( (txn-txp)*16.e-6, (rxn-rxp)*16.e-6 )

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


    def getDate(self):
        self.page.set(0)
        v = self.DateBlock.get()
        def toChar(sh,w=v):
            return (w>>(32*sh))&0xff

        r = '{:c}{:c}/{:c}{:c}/20{:c}{:c}'.format(toChar(2),toChar(3),toChar(4),toChar(5),toChar(0),toChar(1))
        return r

    def getRxPwr(self):  #mW
        self.page.set(0)
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
        self.page.set(0)
        v = self.TxBiasBlock.get()

        def word(a,o):
            return (a >> (32*o))&0xff
        def tou16(a,o):
            return struct.unpack('H',struct.pack('BB',word(a,o+1),word(a,o)))[0]
        def pwr(lane,v=v):
            p = tou16(v,2*lane)
            return p * 0.002
                
        return (pwr(0),pwr(1),pwr(2),pwr(3))


class I2cBus(pr.Device):
    def __init__(self,
                 name        = 'I2cBus',
                 description = 'Local bus',
                 **kwargs):
        super().__init__(
            name        = name,
            description = description,
            **kwargs
        )

        self.add(pr.RemoteVariable(
            name      = 'select',
            offset    = 0x0,
            bitSize   = 8,
            mode      = 'RW'
        ))

        self.add(QSFPMonitor(
            name   = 'QSFP0',
            offset = 0x400
        ))

        self.add(QSFPMonitor(
            name   = 'QSFP1',
            offset = 0x800
        ))

    def selectDevice(self, device):
        idev = 0
        if 'QSFP0' in device:
            idev |= (1<<4)
        if 'QSFP1' in device:
            idev |= (1<<1)
        if 'SI570' in device:
            idev |= (1<<2)
        self.select.set(idev)

class Top(pr.Device):

    def __init__(   self,       
            name        = "KCU",
            description = "Container for KCU",
            memBase     = 0,
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)
        
        self.add(TDetSemi( 
            memBase = memBase,
            offset  = 0x00A00000, 
            expand  = False,
        ))

        self.add(TDetTiming( 
            memBase = memBase,
            offset  = 0x00C00000, 
            expand  = False,
        ))

        self.add(I2cBus( 
            memBase = memBase,
            offset  = 0x00E00000, 
            expand  = False,
        ))

