import sys
import logging
import struct
import socket

from p4p.nt import NTScalar
from p4p.server import Server, StaticProvider
from p4p.server.thread import SharedPV
from p4p import Value, Type

import argparse

prefix = ''
pvdb = {}     # start with empty dictionary


class DefaultPVHandler(object):
    type = None

    def __init__(self, parent):
        self.parent = parent

    def put(self, pv, op):
        postedval = op.value()
        postedval['timeStamp.secondsPastEpoch'], postedval['timeStamp.nanoseconds'] = divmod(float(time.time()), 1.0)
        pv.post(postedval)
        op.done()

class PVAServer(object):
    def __init__(self, provider_name):
        self.provider = StaticProvider(provider_name)
        self.prefix = prefix
        self.pvs = {}

        self.addSource('GMD' , '239.255.25.2', 10148, [ ('milliJoulesPerPulse', 'f') ])
        self.addSource('XGMD', '239.255.25.3', 10148, [ ('milliJoulesPerPulse', 'f'), ('POSY', 'f') ])

    def addSource(self, name, addr, port, ntypes):
        ia = socket.inet_aton(addr)
        ha = struct.unpack('I',ia)[0]
        iaddr = socket.ntohl(ha)
        print('addr [{:x}]'.format(iaddr))

        addrpv = SharedPV(initial=NTScalar('I').wrap({'value' : iaddr}),
                      handler=DefaultPVHandler(self))
        self.provider.add(f'DAQ:SRCF:{name}:ADDR',addrpv)

        portpv = SharedPV(initial=NTScalar('I').wrap({'value' : port}),
                      handler=DefaultPVHandler(self))
        self.provider.add(f'DAQ:SRCF:{name}:PORT',portpv)

        nid = '1'
        nvalues = { n[0]:0 for n in ntypes }

        pv = SharedPV(initial=Value(Type([('BldPayload',
                                           ('S',None,ntypes))],
                                         id='epics:nt/NTScalar:1.0')),
                      handler=DefaultPVHandler(self))

        self.provider.add(f'DAQ:SRCF:{name}:PAYLOAD',pv)
        self.pvs[name] = (addrpv, portpv, pv)

    def forever(self):
        Server.forever(providers=[self.provider])


def main():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description='host PVs for GMD BLD')

    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    server = PVAServer(__name__)

    try:
        # process PVA transactions
        server.forever()
    except KeyboardInterrupt:
        print('\nInterrupted')

