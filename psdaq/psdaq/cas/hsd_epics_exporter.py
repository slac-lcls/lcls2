#!/usr/bin/env python

import os
import sys
import time
import argparse
import logging
import itertools
import socket
from p4p.client.thread import Context
from p4p.nt.scalar import ntint
from prometheus_client import start_http_server
from prometheus_client.core import GaugeMetricFamily, REGISTRY

PROM_PORT_BASE = 9200
MAX_PROM_PORTS = 100

bases = {'tmo':[f'DAQ:TMO:HSD:1_{dev[0]}:{dev[1]}' for dev in itertools.product(('01','1A','1B','3D','3E','88','89','B1','B2','DA'),('A','B'))],
         'rix':[f'DAQ:RIX:HSD:1_{dev[0]}:{dev[1]}' for dev in itertools.product(('1A','1B','3D','3E'),('A','B'))],}

class CustomCollector():
    def __init__(self, provider, hutch, identifier):
        self.ctx    = Context(provider)
        self._hutch = hutch
        self._id    = identifier
        #  PVs to monitor
        self._monpgp  = []
        self._monjesd = []
        self._monoor  = []
        self._channels = bases[hutch]
        for i in self._channels:
            self._monpgp .append(f'{i}:MONPGP')
            self._monjesd.append(f'{i}:MONJESD')
            self._monoor .append(f'{i}:FEXOOR')

    def collect(self):

        def monpgp_field(pv,field):
            v = getattr(pv,field)
            r = 0
            for b,bv in enumerate(v):
                r |= bv<<b
            return r

        def monjesd_bitfield(pv,field):
            # array elements are ordered as 14 values for lane 0, 15 values for lane 1, ...
            # the 14 values are [GTResetDone, RxDataValid, RxDataNAlign, SyncStatus, RxBufferOF, RxBufferUF, CommaPos, ModuleEn, SysRefDet, CommaDet, DispErr, DecErr, BuffLatency, CdrStatus]
            r = 0
            for i in range(8):
                r |= pv.stat[i*15+field]<<i
            return r

        def monjesd_counter(pv,field):
            # array elements are ordered as 14 values for lane 0, 15 values for lane 1, ...
            # the 14 values are [GTResetDone, RxDataValid, RxDataNAlign, SyncStatus, RxBufferOF, RxBufferUF, CommaPos, ModuleEn, SysRefDet, CommaDet, DispErr, DecErr, BuffLatency, CdrStatus]
            r = 0
            for i in range(8):
                r += pv.stat[i*15+field]
            return r

        #  Use this generator to fetch the PVs and parse the structure to compute bitmasks
        #  return name, values
        def get_metrics():
            try:
                values = self.ctx.get(self._monpgp)
                fields = []
                for v in values:
                    fields.append( monpgp_field(v,'loclinkrdy') )
                yield ('loclinkrdy', fields)

                fields = []
                for v in values:
                    fields.append( monpgp_field(v,'remlinkrdy') )
                yield ('remlinkrdy', fields)
            except TimeoutError:
                logging.debug('collect %s: TimeoutError' % self._monpgp)

            try:
                values = self.ctx.get(self._monjesd)
                fields = []
                for v in values:
                    fields.append( monjesd_bitfield(v,2) )  # RxDataNAlign
                yield ('RxDataNAlign', fields)
                counts = []
                for v in values:
                    counts.append( monjesd_counter(v,14) )  # StatusCnt
                yield ('RxStatusCnt', counts)
            except TimeoutError:
                logging.debug('collect %s: TimeoutError' % self._monjesd)

            try:
                values = self.ctx.get(self._monoor)
                fields = []
                for v in values:
                    # strangely, ctx.get return unwrapped values first time, then wrapped values
                    fields.append(v.value if hasattr(v,'value') else v)
                yield ('fexoor', fields)
            except TimeoutError:
                logging.debug('collect %s: TimeoutError' % self._monoor)

        for name, fields in get_metrics():
            g = GaugeMetricFamily(name, documentation='', labels=['instrument','channel','id'])
            for i,field in enumerate(fields):
                g.add_metric([self._hutch,self._channels[i],self._id], field)
                yield g


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
    parser = argparse.ArgumentParser(prog=sys.argv[0], description='host PVs for HSDs')

    parser.add_argument('-H', required=False, help='e.g. tst', metavar='HUTCH', default='tst')
    parser.add_argument('-M', required=False, help='Prometheus config file directory', metavar='PROMETHEUS_DIR', default='')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # Start up the server to expose the metrics.
    i = args.H
    c = CustomCollector('pva', args.H, i)
    REGISTRY.register(c)
    c.collect()
    #start_http_server(9200)
    rc = createExposer(args.M)
    while rc:
        time.sleep(5)

if __name__ == '__main__':
    main()
