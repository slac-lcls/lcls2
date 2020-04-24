#!/usr/bin/env python

import sys
import time
import argparse
import logging
from p4p.client.thread import Context
from prometheus_client import start_http_server
from prometheus_client.core import GaugeMetricFamily, REGISTRY

class CustomCollector():
    def __init__(self):
        self.pvactx = Context('pva')
        self._pvs   = {}

    def registerPV(self, name, pv, hutch):
        self._hutch = hutch
        pvs = []
        for i in range(8):
            pvs.append( pv%i )
        self._pvs[name] = pvs

    def collect(self):
        for name,pvs in self._pvs.items():
            g = GaugeMetricFamily(name, documentation='', labels=['instrument','partition'])
            values = self.pvactx.get(pvs)
            logging.debug('collect %s: %s' % (pvs, str(values)))
            for i in range(8):
                g.add_metric([self._hutch,str(i)], values[i].raw.value)
            yield g

def main():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description='host PVs for XPM')

    parser.add_argument('-H', required=False, help='e.g. tst', metavar='HUTCH', default='tst')
    parser.add_argument('-P', required=True, help='e.g. DAQ:LAB2:XPM:2', metavar='PREFIX')
    parser.add_argument('N', help='e.g. DeadFrac', nargs='+', metavar='NAME')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Start up the server to expose the metrics.
    c = CustomCollector()
    REGISTRY.register(c)
    for name in args.N:
        c.registerPV(name, args.P + ':PART:%d:' + name, args.H)
    c.collect()
    start_http_server(9200)
    while True:
        time.sleep(5)

if __name__ == '__main__':
    main()

