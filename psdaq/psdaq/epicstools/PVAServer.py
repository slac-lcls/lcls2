#!/usr/bin/env python
"""
This implements a PVAccess server that takes in a definition of PV's and create a PVAccess server serving those PV's.
This is based on Michael Davidsaver's mailbox_server.py example.
"""

from __future__ import print_function

import time, logging

from p4p.nt import NTScalar
from p4p.server import Server, StaticProvider
from p4p.server.thread import SharedPV
from p4p import Value, Type

logger = logging.getLogger(__name__)

class DefaultPVHandler(object):
    type = None

    def put(self, pv, op):
        logger.debug("Current MDEL %s", pv.current().get('MDEL'))
        # if pv.current().get('MDEL') and abs(pv.current().value() - op.value()) >= pv.current().get('MDEL'):
        postedval = op.value()
        postedval['timeStamp.secondsPastEpoch'], postedval['timeStamp.nanoseconds'] = divmod(float(time.time()), 1.0)
        pv.post(postedval)
        op.done()

__pcastypes2p4ptype__ = {
'int': 'I',
'float': 'f',
'string': 's',
'char': 'B'
}

__pcastypes2startingval__ = {
'int': 0,
'float': 0.0,
'string': '',
'char': 0,
}


class PVAServer(object):
    def __init__(self, provider_name):
        self.provider = StaticProvider(provider_name)
        self.pvs = []

    def createPV(self, prefix, pvdefs):
        """
        Takes a string prefix and a dict of pv definitions similar to pcaspy and creates PVAccess pv's for them.
        Example PV definitions:
        {'type' : 'int', 'count' : 2, 'value' : [0,0x0fffffff] }
        {'type' : 'float', 'value' : 156.25 }
        """
        for name, pvdef in pvdefs.items():
            logger.debug("Creating PV for %s", prefix+name)
            missing_specs = pvdef.keys() - set(['type', 'count', 'value', 'extra'])
            if missing_specs:
                raise Exception("Do not have support for specifier {0} as of yet".format(",".join(missing_specs)))
            try:
                tp = __pcastypes2p4ptype__[pvdef['type']]
                starting_val = pvdef.get('value', __pcastypes2startingval__[pvdef['type']])
                if pvdef.get('count', 1) > 1:
                    tp = 'a' + tp
                    starting_val = pvdef.get('value', [__pcastypes2startingval__[pvdef['type']]] * pvdef['count'])
                init_val = {"value": starting_val}
                extra_defs = []
                if 'extra' in pvdef:
                    init_val.update({fn : fv for (fn, _, fv) in pvdef['extra']})
                    extra_defs = [ (fn, __pcastypes2p4ptype__[ft]) for (fn, ft, _) in pvdef['extra'] ]
                    logger.debug("NTScalar(%s, extra=%s).wrap(%s)", tp, extra_defs, init_val)
                pv = SharedPV(initial=NTScalar(tp, extra=extra_defs).wrap(init_val), handler=DefaultPVHandler())
            except:
                pv = SharedPV(initial=Value(Type(pvdef['type']),pvdef['value']), handler=DefaultPVHandler())
            self.provider.add(prefix + name, pv)
            self.pvs.append(pv) # we must keep a reference in order to keep the Handler from being collected
            logger.debug("Created PV for %s", prefix+name)

    def forever(self):
        Server.forever(providers=[self.provider])

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    srv = PVAServer("Testing123")
    srv.createPV("Testing123", {
        "One":   {'type' : 'int', 'count' : 2, 'value' : [0,0x0fffffff] },
        "Two":   {'type' : 'float', 'value' : 156.25 },
        "Three": {'type' : 'string', 'value' : "Hello World" },
    })
    srv.forever()
