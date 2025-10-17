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
MAX_PROM_PORTS = 100

class GroupCollector():
    def __init__(self):
        self._collectors = []

    def add(self,c):
        self._collectors.append(c)

    def collect(self):
        for c in self._collectors:
            for d in c.collect():
                yield d

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
    for i in range(MAX_PROM_PORTS):
        port = PROM_PORT_BASE + i
        try:
            start_http_server(port)
            fileName = f'{prometheusDir}/drpmon_{hostname}_{i}.yaml'
            # Commented out the existing file check so that the file's date is refreshed
            if True: #not os.path.exists(fileName):
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
    logging.error('No available port found for providing run-time monitoring')
    return False

def main():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description='host PVs for XPM')

    parser.add_argument('-H', required=False, help='e.g. tst', metavar='HUTCH', default='tst')
    parser.add_argument('-M', required=False, help='Prometheus config file directory', metavar='PROMETHEUS_DIR', default='')
    parser.add_argument('-I', required=False, help='e.g. xpm-2', metavar='ID')
    parser.add_argument('-E', required=False, help='pva or ca', metavar='EPICS_PROVIDER', default='pva') # p4p allows only 'pva'?
    parser.add_argument('-P', required=False,  help='e.g. DAQ:LAB2:XPM:2', metavar='PREFIX', default=None)
    parser.add_argument('-D', required=False,  help='semi-colon separated list of hutch,id,PV triplets', metavar='TRIPLET', default=None)
    parser.add_argument('-G', required=False, help='Global PVs like CuTiming:CrcErrs', metavar='GLOBALS', default='')
    parser.add_argument('N',  help='e.g. DeadFrac', nargs='*', metavar='NAME')
    parser.add_argument('-t', '--test', action='store_true', help='test only')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # Start up the server to expose the metrics.
    i = args.I if args.I is not None else args.H  # Default to previous behavior if -I is not given

    d = []
    if args.D is None:
        d.append((args.H, i, args.P))
    else:
        print(f'args.D {args.D}')
        for w in args.D.split(','):
            print(f'w {w}')
            q = w.split('/')
            if len(q)==3:
                d.append(q)
            else:
                raise ValueError(f'Error parsing -D {args.D}.  Too few comma-separated ({len(q)}) entries in a triplet.')
        
    group = GroupCollector()
    for entry in d:
        print(f'Forming CustomCollector with {entry}')
        c = CustomCollector(args.E, entry[0], entry[1])

        pv = entry[2]
        if args.G:
            for name in args.G.split(','):
                c.registerGlobalPV(name, pv + ':' + name)

        if args.N is not None:
            for name in args.N:
                c.registerPV(name, pv + ':PART:%d:' + name)

        group.add(c)

    REGISTRY.register(group)
    #start_http_server(9200)
    if not args.test:
        rc = createExposer(args.M)
        while rc:
            time.sleep(5)

if __name__ == '__main__':
    main()
