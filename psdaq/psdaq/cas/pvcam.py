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

        #  Monitoring
        self.image     = SharedPV(initial=NTScalar('ai').wrap({'value':[0]*1024*1024}),
                                  handler=DefaultPVHandler())

        self.provider.add(prefix, self.image)

    def forever(self):
        Server.forever(providers=[self.provider])


import argparse

def main():

    parser = argparse.ArgumentParser(prog=sys.argv[0], description='host PVs for High Speed Digitizer')

    parser.add_argument('-P', required=True, help='DAQ:LAB2:HSD:DEV06_3E:A', metavar='PREFIX')
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
