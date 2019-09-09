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

logger = logging.getLogger(__name__)

class DefaultPVHandler(object):
    type = None

    def __init__(self, parent):
        self.parent = parent

    def put(self, pv, op):
        postedval = op.value()
        postedval['timeStamp.secondsPastEpoch'], postedval['timeStamp.nanoseconds'] = divmod(float(time.time()), 1.0)
        pv.post(postedval)
        op.done()
        self.parent.update()

class PVAServer(object):
    def __init__(self, provider_name, prefix):
        self.provider = StaticProvider(provider_name)
        self.prefix = prefix
        self.pvs = []

        self.fieldNames = SharedPV(initial=NTScalar('as').wrap
                                   ({'value' : ['pid%02x'%i for i in range(31)]}),
                                   handler=DefaultPVHandler(self))

        # 'i' (integer) or 'f' (float)
        self.fieldTypes = SharedPV(initial=NTScalar('aB').wrap({'value' : [ord('i')]*31}),
                                   handler=DefaultPVHandler(self))

        self.fieldMask  = SharedPV(initial=NTScalar('I').wrap({'value' : 15}),
                                   handler=DefaultPVHandler(self))

        self.payload    = SharedPV(initial=Value(Type([]),{}), 
                                   handler=DefaultPVHandler(self))

        self.provider.add(prefix+'HPS:FIELDNAMES',self.fieldNames)
        self.provider.add(prefix+'HPS:FIELDTYPES',self.fieldTypes)
        self.provider.add(prefix+'HPS:FIELDMASK' ,self.fieldMask)
        self.provider.add(prefix+'PAYLOAD'   ,self.payload)

    def update(self):
        mask  = self.fieldMask .current().get('value')
        names = self.fieldNames.current().get('value')
        types = self.fieldTypes.current().get('value')
        oid   = self.payload   .current().getID()
        nid   = str(mask)

        if nid==oid:
            nid += 'a'
        ntypes  = []
        nvalues = {}
        ntypes.append( ('valid', 'i') )
        nvalues[ 'valid' ] = 0
        for i in range(31):
            if mask&1:
                ntypes.append( (names[i], chr(types[i])) )
                nvalues[ names[i] ] = 0
            mask >>= 1

        pvname = self.prefix+'PAYLOAD'
        self.provider.remove(pvname)
        self.payload = SharedPV(initial=Value(Type(ntypes,id=nid),nvalues), handler=DefaultPVHandler(self))
        print('Payload struct ID %s'%self.payload.current().getID())
        self.provider.add(pvname,self.payload)

    def forever(self):
        Server.forever(providers=[self.provider])


import argparse

def main():
    global prefix
    prefix = ''

    parser = argparse.ArgumentParser(prog=sys.argv[0], description='host PVs for HPS diagnostic bus')

    parser.add_argument('-P', required=True, help='DAQ:SIM', metavar='PREFIX')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    stationstr = ''
    prefix = args.P+':'

    server = PVAServer(__name__, args.P+':')

    try:
        # process PVA transactions
        server.forever()
    except KeyboardInterrupt:
        print('\nInterrupted')



if __name__ == '__main__':
    main()
