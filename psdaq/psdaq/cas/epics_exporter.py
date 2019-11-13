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
        self._name = None
        self._pv = None

    def registerPV(self, name, pv):
        self._name = name
        self._pv = pv

    def collect(self):
        if self._pv is None:
            return
        g = GaugeMetricFamily(self._name, documentation='', labels=['partition'])
        for p in range(8):
            pv = self._pv % p
            value = self.pvactx.get(pv)
            logging.debug('collect %s: %s' % (pv, str(value.raw.value)))
            g.add_metric([str(p)], value.raw.value)
        yield g

def main():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description='host PVs for XPM')

    parser.add_argument('-P', required=True, help='e.g. DAQ:LAB2:XPM:2', metavar='PREFIX')
    parser.add_argument('-N', required=True, help='e.g. DeadFrac', metavar='NAME')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Start up the server to expose the metrics.
    c = CustomCollector()
    REGISTRY.register(c)
    c.registerPV(args.N, args.P + ':PART:%d:' + args.N)
    c.collect()
    start_http_server(9200)
    while True:
        time.sleep(5)

if __name__ == '__main__':
    main()

