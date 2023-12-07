#!/usr/bin/env python3

import os
import sys
import time
import jmespath
import requests
import argparse
import logging
import cothread
from p4p.nt import NTScalar
from p4p.server import Server, StaticProvider
from p4p.server.cothread import SharedPV
import json
from pathlib import Path

import pprint

_log = logging.getLogger(__name__)


class PromMetric(object):
    def __init__(self, srvurl, query):
        self._srvurl = srvurl
        self._query  = query

    def query(self):
        payload = {'query': self._query}
        url = f'{self._srvurl}/api/v1/query'
        response = requests.get(url, params=payload)
        #_log.debug(f"query response: {response.status_code}")
        response.raise_for_status()

        data = response.json()
        #print('query response: ')
        #pprint.pprint(data)
        return data

    def get(self):
        result = self.query()
        if 'warnings' in result:
            _log.warning(f"Warnings from query {self._query}:")
            for warning in result['warnings']:
                _log.warning(warning)
        if result['status'] == 'success':
            if result['data']['resultType'] == 'vector':
                if len(result['data']['result']) == 1:
                    return result['data']['result'][0]['value']
                else:
                    raise RuntimeError(f"More than 1 item ({len(result['data']['result'])}) from query {self._query}")
            elif result['data']['resultType'] in ['scalar','string']:
                return result['data']['result']['value']
            else:
                raise RuntimeError(f"Unsupported result type {result['data']['resultType']} from query {self._query}")
        else:
            _log.error(f"Error type {result['errorType']}")
            _log.error(f"Error {result['error']}")
            raise RuntimeError(f"Error from query {self._query}")

class Timer(object):
    def __init__(self, interval):
        self._interval = interval
        self._timer = None
        self._callbacks = []
        self._active = False

    def append(self, callback):
        if self._timer is None:
            self._timer = cothread.Timer(self._interval, self._tick, retrigger=True)
        if callback not in self._callbacks:
            self._callbacks.append(callback)
        self._active = True

    def _tick(self):
        if not self._active:
            _log.debug("Close")

            # cancel timer until a new first client arrives
            self._timer.cancel()
            self._timer = None
        else:
            for callback in self._callbacks:
                callback()

    def remove(self, callback):
        if self._timer is not None:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
            if len(self._callbacks) == 0:
                self._active = False

class Handler(object):
    def __init__(self, timer, name, metric, typeCode, alarm):
        self._timer = timer
        self._name = name
        self._metric = metric
        self._NT = NTScalar(typeCode, valueAlarm=alarm is not None)
        self._valueAlarm = alarm
        self._pv = None
        self._active = False
        self._eq = cothread.EventQueue()

    def onFirstConnect(self, pv):
        _log.debug(f"First client connects to {self._name}")
        self._timer.append(self._tick)
        self._pv = pv
        self._active = True

    def _tick(self):
        if not self._active:
            _log.debug(f"Closing {self._name}")
            # no clients connected
            if self._pv.isOpen():
                self._pv.close()

            # remove handling by timer until a new first client arrives
            self._timer.remove(self._tick)
            self._pv = None

        else:
            timestamp, value = self._metric.get()
            wrapped = self._NT.wrap(value, timestamp=timestamp)

            if not self._pv.isOpen():
                _log.debug(f"Open {self._name} [{timestamp}, {value}]")
                for key in self._valueAlarm.keys():
                    wrapped['valueAlarm.'+key] = self._valueAlarm[key]
                self._pv.open(wrapped)

            else:
                _log.debug(f"Tick {self._name} [{timestamp}, {value}]")
                while len(self._eq):
                    (item, value) = self._eq.Wait(0)
                    wrapped[item] = value
                self._pv.post(wrapped)

    def onLastDisconnect(self, pv):
        _log.debug(f"Last client disconnects from {self._name}")
        # mark in-active, but don't immediately close()
        self._active = False

    def put(self, pv, op):
        msg = ''
        for item in op.value().changedSet(expand=False):
            if item.split('.')[0] == 'valueAlarm':
                self._valueAlarm[item.split('.')[1]] = op.value()[item]
                self._eq.Signal( (item, op.value()[item]) )
            else:
                msg += ' ' + item
        op.done(error=None if len(msg) == 0 else ('Put not supported:' + msg))


def main():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description='Prometheus metric PV server')

    srv = 'http://psmetric03:9090'

    parser.add_argument('-J', '--job',    required=False, metavar='JOB',            default='psdm', help='e.g. psdm')
    parser.add_argument('-H', '--inst',   required=True,  metavar='HUTCH',          default='tst',  help='e.g. tst')
    parser.add_argument('-p', '--part',   required=True,  metavar='PARTITION',      default='0',    help='e.g. 0')
    parser.add_argument('-S',             required=False, metavar='PROMETHEUS_SRV', default=srv,    help=f'Prometheus server [{srv}]')
    parser.add_argument('-P',             required=False, metavar='PREFIX',         default='DAQ',  help='e.g. DAQ:LAB2:XPM:2')
    parser.add_argument('-I', type=float, required=False, metavar='INTERVAL',       default='1',    help='Sampling interval (s)')
    parser.add_argument('filename',                       metavar='FILENAME',                       help='Path to json configuration file')
    parser.add_argument('-v', '--verbose', action='store_true',                                     help='be verbose')

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    promserver = os.environ.get('DM_PROM_SERVER', args.S)

    config = json.loads(Path(args.filename).read_text().replace('%J', args.job).replace('%I', args.inst).replace('%P', args.part))
    if args.verbose:  pprint.pprint(config)

    metrics = {}
    pvs = {}
    timer = Timer(args.I)
    prefix = f'{args.inst}:{args.part}:'
    if args.P is not None:  prefix = f'{args.P}:' + prefix
    for item in config['metrics']:
        metric = PromMetric(args.S, item['query'])
        typeCode = item['type'] if 'type' in item else 'd'
        alarm = item['alarm'] if 'alarm' in item else None
        pvs[prefix+item['name']] = SharedPV(handler=Handler(timer, prefix+item['name'], metric, typeCode, alarm))

    with Server(providers=[pvs]):
        try:
            cothread.WaitForQuit()
        except KeyboardInterrupt:
            _log.info('\nInterrupted')

    _log.info('Exiting')


if __name__ == '__main__':
    main()
