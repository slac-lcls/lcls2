#!/usr/bin/env python

import os
import sys
import time
import argparse
import logging
import socket
from p4p.client.thread import Context
from prometheus_client import start_http_server
from prometheus_client.core import GaugeMetricFamily, REGISTRY

PROM_PORT_BASE = 9200

class CustomCollector():
    def __init__(self, provider, hutch, identifier):
        self.ctx    = Context(provider)
        self._hutch = hutch
        self._id    = identifier
        self._glob  = {}
        self._pvs   = {}

    def registerGlobalPV(self, name, pv):
        self._glob[name] = pv

    def registerPV(self, name, pv):
        pvs = []
        for i in range(8):
            pvs.append( pv%i )
        self._pvs[name] = pvs

    def collect(self):
        for name,pvs in self._glob.items():
            g = GaugeMetricFamily(name, documentation='', labels=['instrument','id'])
            try:
                values = self.ctx.get(pvs)
                logging.debug('collect %s: %s' % (pvs, str(values)))
                g.add_metric([self._hutch, self._id], values.raw.value)
                yield g
            except TimeoutError:
                logging.debug("collect %s: TimeoutError" % (pvs))

        for name,pvs in self._pvs.items():
            g = GaugeMetricFamily(name, documentation='', labels=['instrument','partition','id'])
            try:
                values = self.ctx.get(pvs)
                logging.debug('collect %s: %s' % (pvs, str(values)))
                for i in range(8):
                    g.add_metric([self._hutch,str(i),self._id], values[i].raw.value)
                    yield g
            except TimeoutError:
                logging.debug("collect %s: TimeoutError" % (pvs))

def createExposer(prometheusDir):
    if prometheusDir == '':
        logging.warning('Unable to update Prometheus configuration: directory not provided')
        return

    hostname = socket.gethostname()
    port = PROM_PORT_BASE
    while port < PROM_PORT_BASE + 100:
        try:
            start_http_server(port)
            fileName = f'{prometheusDir}/drpmon_{hostname}_{port - PROM_PORT_BASE}.yaml'
            if not os.path.exists(fileName):
                try:
                    with open(fileName, 'wt') as f:
                        f.write(f'- targets:\n    - {hostname}:{port}\n')
                except Exception as ex:
                    logging.error(f'Error creating file {fileName}: {ex}')
                    return False
            else:
                pass            # File exists; no need to rewrite it
            logging.info(f'Providing run-time monitoring data on port {port}')
            return True
        except OSError:
            pass                # Port in use
        port += 1
    logging.error('No available port found for providing run-time monitoring')
    return False

def main():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description='host PVs for XPM')

    parser.add_argument('-H', required=False, help='e.g. tst', metavar='HUTCH', default='tst')
    parser.add_argument('-M', required=False, help='Prometheus config file directory', metavar='PROMETHEUS_DIR', default='')
    parser.add_argument('-I', required=False, help='e.g. xpm-2', metavar='ID')
    parser.add_argument('-E', required=False, help='pva or ca', metavar='EPICS_PROVIDER', default='pva') # p4p allows only 'pva'?
    parser.add_argument('-P', required=True,  help='e.g. DAQ:LAB2:XPM:2', metavar='PREFIX')
    parser.add_argument('-G', required=False, help='Global PVs like CuTiming:CrcErrs', metavar='GLOBALS', default='')
    parser.add_argument('N',  help='e.g. DeadFrac', nargs='*', metavar='NAME')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # Start up the server to expose the metrics.
    i = args.I if args.I is not None else args.H  # Default to previous behavior if -I is not given
    c = CustomCollector(args.E, args.H, i)
    REGISTRY.register(c)
    if args.G:
        for name in args.G.split(','):
            c.registerGlobalPV(name, args.P + ':' + name)
    if args.N is not None:
        for name in args.N:
            c.registerPV(name, args.P + ':PART:%d:' + name)
    c.collect()
    #start_http_server(9200)
    rc = createExposer(args.M)
    while rc:
        time.sleep(5)

if __name__ == '__main__':
    main()
