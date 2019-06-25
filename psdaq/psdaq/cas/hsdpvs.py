#!/usr/bin/env python
"""
This implements a PVAccess server that defines the PVs for a HPS server diagnostic bus for conversion to BLD
Any change to the diagnostic bus description (field names, mask) results in an updated payload structure
"""
import sys
import logging
import time

from p4p.nt import NTScalar
from p4p.server import Server, StaticProvider
from p4p.server.thread import SharedPV
from p4p import Value, Type

from psdaq.hsd.pvdef import *

#import pyrogue as pr

logger = logging.getLogger(__name__)

# pv_fieldname, pv_fieldtype, default_value, mmap_address, bit_size, bit_shift

def pvTypes(nt):
    result = []
    for x,y in nt.items():
        result.append((x,y[0]))
    return result

def pvValues(nt):
    result = []
    for x,y in nt.items():
        result.append((x,y[1]))
    return result

def MySharedPV(nt,cb=None):
    return SharedPV(initial=Value(Type(pvTypes(nt)),pvValues(nt)),
                    handler=DefaultPVHandler(cb))

class DefaultPVHandler(object):
    type = None

    def __init__(self,callback=None):
        self.callback = callback

    def put(self, pv, op):
        postedval = op.value()
#        postedval['timeStamp.secondsPastEpoch'], postedval['timeStamp.nanoseconds'] = divmod(float(time.time()), 1.0)
        pv.post(postedval)
        op.done()
        if self.callback is not None:
            self.callback(postedval)

class ChipServer(object):
    def __init__(self, provider, prefix):
        self.provider = provider
        self.prefix = prefix

        #  Make configuration one PV access for each readout channel
        self.daqConfig      = MySharedPV(daqConfig,self.updateDaqConfig)
        self.ready          = SharedPV(initial=NTScalar('I').wrap({'value' : 0}),
                                       handler=DefaultPVHandler())

        self.provider.add(prefix+':CONFIG',self.daqConfig)
        self.provider.add(prefix+':READY' ,self.ready)

        #  Monitoring
        self.fwBuild     = SharedPV(initial=NTScalar('s').wrap({'value':''}),
                                    handler=DefaultPVHandler())
        self.monTiming   = MySharedPV(monTiming)
        self.monPgp      = MySharedPV(monPgp)
        self.monRawBuf   = MySharedPV(monBuf)
        self.monFexBuf   = MySharedPV(monBuf)
        self.monRawDet   = MySharedPV(monBufDetail)
        self.monEnv      = MySharedPV(monEnv)
        self.monAdc      = MySharedPV(monAdc)
        self.monJesd     = MySharedPV(monJesd)
        self.monJesdTtl  = MySharedPV(monJesdTtl)

        self.provider.add(prefix+':FWBUILD'   ,self.fwBuild)
        self.provider.add(prefix+':MONTIMING' ,self.monTiming)
        self.provider.add(prefix+':MONPGP'    ,self.monPgp)
        self.provider.add(prefix+':MONRAWBUF' ,self.monRawBuf)
        self.provider.add(prefix+':MONFEXBUF' ,self.monFexBuf)
        self.provider.add(prefix+':MONRAWDET' ,self.monRawDet)
        self.provider.add(prefix+':MONENV'    ,self.monEnv)
        self.provider.add(prefix+':MONADC'    ,self.monAdc)
        self.provider.add(prefix+':MONJESD'   ,self.monJesd)
        self.provider.add(prefix+':MONJESDTTL',self.monJesdTtl)

    def updateDaqConfig(self,value):
        pass

class PVAServer(object):
    def __init__(self, provider_name, prefix):
        self.provider = StaticProvider(provider_name)
        self.a = ChipServer(self.provider, prefix+':A')
        self.b = ChipServer(self.provider, prefix+':B')

    def forever(self):
        Server.forever(providers=[self.provider])


#
#  Brief rogue implementation of HSD
#
#class Top(pr.Root):
#    def __init__(self, 
#                 name        = "Top",
#                 description = "Container for HSD",
#                 dev         = '/dev/pcie_adc_3e',
#                 hwType      = 'special',
#                 lane        = 0,
#                 **kwargs):
#        super().__init__(name=name, description=description, **kwargs)

        # Connect the SRPv0 to PCIe
#        memMap = rogue.hardware.axi.AxiMemMap(dev)
        
#        pr.busConnect(memMast,memMap)

#        self.add(DaqConfig(name    = 'DaqConfig',
#                           memBase = memMap,
#                           offset  = 0,
#                           expand  = False))

#        self.add(MonTiming(name    = 'MonTiming',
#                           memBase = memMap,
#                           offset  = 0,
#                           expand  = False))

#class DaqConfig(pr.Device):
#    def __init__(self, **kwargs):
#        super().__init__(description='DaqConfig', **kwargs)

        # Add devices
#        for x,y in daqConfig:
#            self.add(pr.RemoteVariable(name        = x,
#                                       description = x,
#                                       offset      = y[2],
#                                       bitSize     = y[3],
#                                       bitOffset   = y[4],
#                                       base        = pr.UInt,
#                                       mode        = 'RW',
#                                       verify      = False,))

#class MonTiming(pr.Device):
#    def __init__(self, **kwargs):
#        super().__init__(description='MonTiming', **kwargs)

#        for x,y in monTiming:
#            self.add(pr.RemoteVariable(name        = x,
#                                       description = x,
#                                       offset      = y[2],
#                                       bitSize     = y[3],
#                                       bitOffset   = y[4],
#                                       base        = pr.UInt,
#                                       mode        = 'RW',
#                                       verify      = False,))

import argparse

def main():

    parser = argparse.ArgumentParser(prog=sys.argv[0], description='host PVs for High Speed Digitizer')

    parser.add_argument('-P', required=True, help='DAQ:LAB2:HSD:DEV06_3E:A', metavar='PREFIX')
#    parser.add_argument('-d', required=True, help='device filename', metavar='DEV')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    server = PVAServer(__name__, args.P)

    try:
        # process PVA transactions
        server.forever()
    except KeyboardInterrupt:
        print('\nInterrupted')



if __name__ == '__main__':
    main()
