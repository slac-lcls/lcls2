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

#import pyrogue as pr

logger = logging.getLogger(__name__)

# pv_fieldname, pv_fieldtype, default_value, mmap_address, bit_size, bit_shift

daqConfig = {'readoutGroup':('i', 0),
             'enable'      :('i', 0),
             'raw_start'   :('i', 4),
             'raw_gate'    :('i', 20),
             'raw_prescale':('i', 1),
             'fex_start'   :('i', 4),
             'fex_gate'    :('i', 20),
             'fex_prescale':('i', 0),
             'fex_ymin'    :('i', 2040),
             'fex_ymax'    :('i', 2056),
             'fex_xpre'    :('i', 1),
             'fex_xpost'   :('i', 1) }

monTiming = {'timframecnt':('i', 0),
             'timpausecnt':('i', 0),
             'trigcnt'    :('i', 0),
             'trigcntsum' :('i', 0),
             'msgdelayset':('i', 0),
             'msgdelayget':('i', 0),
             'headercntl0':('i', 0),
             'headercntof':('i', 0) }


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

class PVAServer(object):
    def __init__(self, provider_name, prefix):
        self.provider = StaticProvider(provider_name)
        self.prefix = prefix

        #  Make configuration one PV access for each readout channel
        self.daqConfigA     = MySharedPV(daqConfig,self.updateDaqConfigA)
        self.daqConfigB     = MySharedPV(daqConfig,self.updateDaqConfigB)
        self.readyA         = SharedPV(initial=NTScalar('I').wrap({'value' : 0}),
                                       handler=DefaultPVHandler())
        self.readyB         = SharedPV(initial=NTScalar('I').wrap({'value' : 0}),
                                       handler=DefaultPVHandler())

        self.provider.add(prefix+':A:CONFIG',self.daqConfigA)
        self.provider.add(prefix+':B:CONFIG',self.daqConfigB)
        self.provider.add(prefix+':A:READY' ,self.readyA)
        self.provider.add(prefix+':B:READY' ,self.readyB)

        #  Monitoring
        self.monTimingA  = MySharedPV(monTiming)
        #self.monTimingB  = SharedPV()
        #self.monPgpA     = SharedPV()
        #self.monPgpB     = SharedPV()
        #self.monFexStatA = SharedPV()
        #self.monFexStatB = SharedPV()
        #self.monJesdA    = SharedPV()
        #self.monJesdB    = SharedPV()
        #self.monEnv      = SharedPV()
        self.provider.add(prefix,self.monTimingA)

    def updateDaqConfigA(self,value):
        pass

    def updateDaqConfigB(self,value):
        pass

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

    parser.add_argument('-P', required=True, help='DAQ:LAB2:HSD:DEV06_3E', metavar='PREFIX')
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
