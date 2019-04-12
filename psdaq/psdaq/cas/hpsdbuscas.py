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
        parent.update()

class PVAServer(object):
    def __init__(self, provider_name, prefix):
        self.provider = StaticProvider(provider_name)
        self.prefix = prefix
        self.pvs = []

        self.fieldNames = SharedPV(initial=NTScalar('as').wrap({'value' : ['FP0_sev0','FP1_sev1','FP2_sev2','FP3_sev3',
                                                                           'FP0-sev0','FP1-sev0','FP2-sev0','FP3-sev0',
                                                                           'PID2','PID3','PID4','PID5','PID6','PID7','PID8','PID9',
                                                                           'C+12','C+13','C+14','C+15','C+16','C+17','C+18','C+19',
                                                                           'C-12','C-13','C-14','C-15','C-16','C-17','C-18',]}),
                                   handler=DefaultPVHandler(self))
        self.provider.add(prefix+'FIELDNAMES',self.fieldNames)

        self.fieldTypes = SharedPV(initial=NTScalar('aB').wrap({'value' : [ord('i')]*31}),  # 'i' (integer) or 'f' (float)
                                   handler=DefaultPVHandler(self))
        self.provider.add(prefix+'FIELDTYPES',self.fieldTypes)

        self.fieldMask = SharedPV(initial=NTScalar('I').wrap({'value' : 0}),
                                   handler=DefaultPVHandler(self))
        self.provider.add(prefix+'FIELDMASK',self.fieldMask)

        self.payload = SharedPV(initial=Value(Type([]),{}), handler=DefaultPVHandler(self))
        self.provider.add(prefix+'PAYLOAD',self.payload)

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
        for i in range(31):
            if mask&1:
                ntypes.append( (names[i], chr(types[i])) )
                nvalues[ names[i] ] = 0
            mask >>= 1
        ntypes.append( ('valid', 'i') )
        nvalues[ 'valid' ] = 0

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
