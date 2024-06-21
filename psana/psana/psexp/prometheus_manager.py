import os
import time
import socket
import getpass
from prometheus_client import CollectorRegistry, Counter, push_to_gateway, Summary, Gauge
from prometheus_client import start_http_server
from prometheus_client.exposition import tls_auth_handler

from psana import utils
from psana.psexp.tools import mode
if mode == 'mpi':
    from mpi4py import MPI
    logger = utils.Logger(myrank=MPI.COMM_WORLD.Get_rank())
else:
    logger = utils.Logger()

PUSH_INTERVAL_SECS= 5

if socket.gethostname().startswith('sdf'):
    PUSH_GATEWAY = 'https://172.24.5.157:9091'   
else:
    PUSH_GATEWAY = 'psdm03:9091'                

PROM_PORT_BASE = 9200      # Used by the http exposer; Value should match DAQ's

HTTP_EXPOSER_STARTED = False

def createExposer(prometheusCfgDir):
    if prometheusCfgDir == '':
        logging.warning('Unable to update Prometheus configuration: directory not provided')
        return

    # Start only one server per session to avoid multiple scrapings per time point
    global HTTP_EXPOSER_STARTED
    if HTTP_EXPOSER_STARTED:
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
            HTTP_EXPOSER_STARTED = True
            return True
        except OSError:
            pass                # Port in use
        port += 1
    logging.error('No available port found for providing run-time monitoring')
    return False

def my_auth_handler(url, method, timeout, headers, data):
    #TODO Certificate and key files are hard-coded
    certfile = '/sdf/group/lcls/ds/ana/data/prom/promgwclient.pem'
    keyfile = '/sdf/group/lcls/ds/ana/data/prom/promgwclient.key'
    return tls_auth_handler(url, method, timeout, headers, data, certfile, keyfile, insecure_skip_verify=True)

class PrometheusManager(object):
    registry = CollectorRegistry()
    metrics ={
            'psana_smd0_read'       : ('Gauge', 'Disk reading rate (MB/s) for smd0'),
            'psana_smd0_rate'       : ('Gauge', 'Processing rate by smd0 (kHz)'),
            'psana_smd0_wait'       : ('Gauge', 'time spent (s) waiting for EventBuilders'),
            'psana_eb_rate'         : ('Gauge', 'Processing rate by eb (kHz)'),
            'psana_eb_wait_smd0'    : ('Gauge', 'time spent (s) waiting for Smd0'),
            'psana_eb_wait_bd'      : ('Gauge', 'time spent (s) waiting for BigData cores'),
            'psana_bd_rate'         : ('Gauge', 'Processing rate by bd (kHz)'),
            'psana_bd_read'         : ('Gauge', 'Disk reading rate (MB/s) for bd'),
            'psana_bd_ana_rate'     : ('Gauge', 'User-analysis rate on bd (kHz)'),
            'psana_bd_wait'         : ('Gauge', 'time spent (s) waiting for EventBuilder'),
            }
    
    def __init__(self, exp, runnum, job=None):
        self.exp = exp
        self.runnum = runnum
        self.username = getpass.getuser()
        self.rank = 0
        if job is None:
            self.job = f'{self.exp}_{self.runnum}_{self.username}'
        else:
            self.job = job

    def set_rank(self, rank):
        self.rank = rank

    def register(self, metric_name):
        self.registry.register(metric_name)

    def push_metrics(self, e):
        # TODO: Certificate is read at every push, find a better way.
        while not e.isSet():
            if socket.gethostname().startswith('sdf'):
                push_to_gateway(PUSH_GATEWAY, job=self.job, grouping_key={'rank': self.rank}, registry=self.registry, handler=my_auth_handler, timeout=None)
            else:
                push_to_gateway(PUSH_GATEWAY, job=self.job, grouping_key={'rank': self.rank}, registry=self.registry, timeout=None)
            logger.debug(f'TS: %s PUSHED EXP:{self.exp} RUN:{self.runnum} USER:{self.username} RANK:{self.rank} {e.isSet()=}')
            time.sleep(PUSH_INTERVAL_SECS)

    def create_exposer(self, prometheus_cfg_dir):
        return createExposer(prometheus_cfg_dir)

    def create_metric(self, metric_name):
        if metric_name in self.metrics:
            metric_type, desc = self.metrics[metric_name]
            if metric_type == 'Counter':
                self.registry.register(Counter(metric_name, desc))
            elif metric_type == 'Summary':
                self.registry.register(Summary(metric_name, desc))
            elif metric_type == 'Gauge':
                self.registry.register(Gauge(metric_name, desc))
        else:
            print(f'Warning: {metric_name} is not found in the list of available prometheus metrics')

    def get_metric(self, metric_name):
        # get metric object from its name
        # NOTE that _created has to be appended to locate the key
        if metric_name in self.registry._names_to_collectors:
            collector = self.registry._names_to_collectors[metric_name]
        else:
            self.create_metric(metric_name)
            collector = self.registry._names_to_collectors[metric_name]
        return collector



