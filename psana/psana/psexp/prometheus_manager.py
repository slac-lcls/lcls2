import os
import time
import socket
from prometheus_client import CollectorRegistry, Counter, push_to_gateway, Summary, Gauge
from prometheus_client import start_http_server

import logging
logger = logging.getLogger(__name__)

PUSH_INTERVAL_SECS  = 5
PUSH_GATEWAY        = 'psdm03:9091'
PROM_PORT_BASE      = 9200      # Used by the http exposer; Value should match DAQ's

registry = CollectorRegistry()
metrics ={
        'psana_smd0_read'       : ('Counter', 'Counting no. of events/batches/MB read by Smd0'),
        'psana_smd0_sent'       : ('Counter', 'Counting no. of events/batches/MB'),
        'psana_smd0_wait_eb'    : ('Summary', 'time spent (s) waiting for EventBuilders'),
        'psana_eb_sent'         : ('Counter', 'Counting no. of events/batches/MB'),
        'psana_eb_filter'       : ('Counter', 'Counting no. of batches and time spent'),
        'psana_eb_wait_smd0'    : ('Summary', 'time spent (s) waiting for Smd0'),
        'psana_eb_wait_bd'      : ('Summary', 'time spent (s) waiting for BigData cores'),
        'psana_bd_read'         : ('Counter', 'Counting no. of events processed by BigData'),
        'psana_bd_just_read'    : ('Summary', 'time spent (s) reading bigdata'),
        'psana_bd_gen_smd_batch': ('Summary', 'time spent (s) creating a batch of smd events'),
        'psana_bd_gen_evt'      : ('Summary', 'time spent (s) creating an evt'),
        'psana_bd_ana'          : ('Counter', 'time spent (s) in analysis fn on                 \
                                    BigData core'),
        'psana_timestamp'       : ('Gauge',   'Uses different labels (e.g. python_init,         \
                                    first_event) to set the timestamp of that stage'),
        'psana_bd_wait_eb'      : ('Summary', 'time spent (s) waiting for EventBuilder'),
        }

for metric_name, (metric_type, desc) in metrics.items():
    if metric_type == 'Counter':
        registry.register(Counter(metric_name, desc, ['unit', 'endpoint']))
    elif metric_type == 'Summary':
        registry.register(Summary(metric_name, desc))
    elif metric_type == 'Gauge':
        registry.register(Gauge(metric_name, desc, ['checkpoint']))

def createExposer(prometheusCfgDir):
    if prometheusCfgDir == '':
        logging.warning('Unable to update Prometheus configuration: directory not provided')
        return

    hostname = socket.gethostname()
    port = PROM_PORT_BASE
    while port < PROM_PORT_BASE + 100:
        try:
            start_http_server(port)
            fileName = f'{prometheusCfgDir}/drpmon_{hostname}_{port - PROM_PORT_BASE}.yaml'
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

class PrometheusManager(object):
    def __init__(self, jobid):
        self.jobid = jobid
        pass

    def register(self, metric_name):
        registry.register(metric_name)

    def push_metrics(self, e, from_whom=''):
        while not e.isSet():
            push_to_gateway(PUSH_GATEWAY, job='psana_pushgateway', grouping_key={'jobid': self.jobid, 'rank': from_whom}, registry=registry, timeout=None)
            logger.debug('TS: %s PUSHED JOBID: %s RANK: %s e.isSet():%s'%(time.time(), self.jobid, from_whom, e.isSet()))
            time.sleep(PUSH_INTERVAL_SECS)

    def create_exposer(self, prometheus_cfg_dir):
        return createExposer(prometheus_cfg_dir)

    @staticmethod
    def get_metric(metric_name):
        # get metric object from its name
        # NOTE that _created has to be appended to locate the key
        if metric_name in registry._names_to_collectors:
            collector = registry._names_to_collectors[metric_name]
        else:
            collector = registry._names_to_collectors[f'{metric_name}_created']
        return collector



