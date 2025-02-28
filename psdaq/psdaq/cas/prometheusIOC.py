#!/usr/bin/env python3

import os
import sys
import time
import jmespath
import requests
import argparse
import logging
from softioc import softioc, builder, alarm
import cothread
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
        #response.raise_for_status()
        if response.status_code != 200:
            _log.error(f"HTTPError {response.status_code} for {url}, {payload}")
            return None

        data = response.json()
        _log.debug(f"Response to query '{self._query}':")
        if _log.getEffectiveLevel() == logging.DEBUG:  pprint.pprint(data)
        return data

    def get(self):
        result = self.query()
        if result is None:
            return None, None
        if 'warnings' in result:
            _log.warning(f"Warnings from query {self._query}:")
            for warning in result['warnings']:
                _log.warning(warning)
        if result['status'] == 'success':
            if result['data']['resultType'] == 'vector':
                if len(result['data']['result']) == 1:
                    return result['data']['result'][0]['value']
                elif len(result['data']['result']) == 0:
                    return None, None
                else:
                    raise RuntimeError(f"Expected 1 item from query '{self._query}'; "
                                       f"got: {len(result['data']['result'])}:\n({result['data']['result']})")
            elif result['data']['resultType'] in ['scalar','string']:
                return result['data']['result']['value']
            else:
                raise RuntimeError(f"Unsupported result type {result['data']['resultType']} from query {self._query}")
        else:
            _log.error(f"Error type {result['errorType']}")
            _log.error(f"Error {result['error']}")
            raise RuntimeError(f"Error from query {self._query}")

class Handler(object):
    def __init__(self, name, metric, typeCode, valueAlarm):
        # See p4p documentation for type definitions
        self._typeCode = typeCode if typeCode is not None else 'd'
        if self._typeCode in 'df':
            self._pv = builder.aIn(name)
        elif self._typeCode in 'bBhHiIlL':
            self._pv = builder.longIn(name)
        else:
            raise RuntimeError(f"Unsupported typeCode '{self._typeCode}' for '{name}'")
        # alarm is not None if config file specifies alarm levels
        self._initAlarm(valueAlarm if valueAlarm is not None else {})
        self._metric = metric
        _log.info(f"Created PV '{self._pv.name}' with query '{self._metric._query}'")

    def _initAlarm(self, valueAlarm):
        self._valueAlarm = { 'active' : False,
                             'lowAlarmLimit' : 0,
                             'lowWarningLimit' : 0,
                             'highWarningLimit' : 0,
                             'highAlarmLimit' : 0,
                             'lowAlarmSeverity' : 0,
                             'lowWarningSeverity' : 0,
                             'highWarningSeverity' : 0,
                             'highAlarmSeverity' : 0,
                             'hysteresis' : 0 }
        for item in valueAlarm:
            if item in self._valueAlarm:
                self._valueAlarm[item] = valueAlarm[item]
            else:
                _log.error(f"Unrecognized valueAlarm keyword '{item}' for {self._pv.name}")

    def _evalAlarm(self, value):
        active = self._valueAlarm['active']
        if not active:  return alarm.NO_ALARM, alarm.UDF_ALARM

        severity = self._valueAlarm['highAlarmSeverity']
        level = self._valueAlarm['highAlarmLimit']
        if severity > 0 and value >= level:
            return severity, alarm.HIHI_ALARM

        severity = self._valueAlarm['lowAlarmSeverity']
        level = self._valueAlarm['lowAlarmLimit']
        if severity > 0 and value <= level:
            return severity, alarm.LOLO_ALARM

        severity = self._valueAlarm['highWarningSeverity']
        level = self._valueAlarm['highWarningLimit']
        if severity > 0 and value >= level:
            return severity, alarm.HIGH_ALARM

        severity = self._valueAlarm['lowWarningSeverity']
        level = self._valueAlarm['lowWarningLimit']
        if severity > 0 and value <= level:
            return severity, alarm.LOW_ALARM

        return alarm.NO_ALARM, alarm.UDF_ALARM

    def process(self):
        timestamp, value = self._metric.get()
        if timestamp is None:  return
        if self._typeCode in 'df':
            value = float(value)
        elif self._typeCode in 'bBhHiIlL':
            value = int(value)
        else:
            raise RuntimeError(f"Invalid typeCode '{self._typeCode}' for '{self._pv.name}'")
        severity, alarm = self._evalAlarm(value)
        self._pv.set(value, timestamp=timestamp, severity=severity, alarm=alarm)


def main():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description='Prometheus metric PV server')

    srv = 'http://psmetric03:9090'

    parser.add_argument('-J', '--job',    required=False, metavar='JOB',            default='drpmon', help='e.g. drpmon')
    parser.add_argument('-H', '--inst',   required=True,  metavar='HUTCH',          default='tst',    help='e.g. tst')
    parser.add_argument('-p', '--part',   required=True,  metavar='PARTITION',      default='0',      help='e.g. 0')
    parser.add_argument('-S',             required=False, metavar='PROMETHEUS_SRV', default=srv,      help=f'Prometheus server [{srv}]')
    parser.add_argument('-P',             required=False, metavar='PREFIX',         default='DAQ',    help='e.g. DAQ:LAB2')
    parser.add_argument('-I', type=float, required=False, metavar='INTERVAL',       default='1',      help='Sampling interval (s)')
    parser.add_argument('-x', type=int,   required=False, metavar='XPM',            default='0',      help='master XPM')
    parser.add_argument('filename',                       metavar='FILENAME',                         help='Json metric description file')
    parser.add_argument('-v', '--verbose', action='store_true',                                       help='be verbose')

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    promserver = os.environ.get('DM_PROM_SERVER', args.S)

    config = json.loads(Path(args.filename).read_text().replace('%J', args.job)
                                                       .replace('%I', args.inst)
                                                       .replace('%X', f"xpm-{args.x}")
                                                       .replace('%P', args.part))
    if args.verbose:  pprint.pprint(config)

    # Set the record prefix
    prefix = f'{args.P}:{args.inst}:{args.part}'
    builder.SetDeviceName(prefix)

    # Create the records
    handlers = []
    for item in config['metrics']:
        # Treat all items not having a "name" element as comments
        if 'name' not in item:  continue
        metric = PromMetric(promserver, item['query'])
        typeCode = item['type'] if 'type' in item else None
        alarm = item['alarm'] if 'alarm' in item else None
        handlers.append(Handler(item['name'], metric, typeCode, alarm))

    # Boilerplate get the IOC started
    builder.LoadDatabase()
    softioc.iocInit()

    # Start processes required to be run after iocInit
    def update():
        while True:
            for handler in handlers:
                handler.process()
            cothread.Sleep(args.I)

    cothread.Spawn(update)

    # Finally leave the IOC running with an interactive shell.
    #softioc.interactive_ioc(globals())

    # Finally leave the IOC running
    try:
        cothread.WaitForQuit()
    except KeyboardInterrupt:
        _log.info('\nInterrupted')

    _log.info('Exiting')


if __name__ == '__main__':
    main()
