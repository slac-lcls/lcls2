import getpass
import os
import socket
import time
from subprocess import Popen

from prometheus_client import (CollectorRegistry, Counter, Gauge, Summary,
                               push_to_gateway, start_http_server)
from prometheus_client.exposition import tls_auth_handler

from psana import utils
from psana.psexp.tools import mode

if mode == "mpi":
    from mpi4py import MPI
    size = MPI.COMM_WORLD.Get_size()
else:
    size = 1

PUSH_INTERVAL_SECS = 5

if socket.gethostname().startswith("sdf"):
    PUSH_GATEWAY = "https://172.24.5.157:9091"
else:
    PUSH_GATEWAY = "psdm03:9091"

PROM_PORT_BASE = 9200  # Used by the http exposer; Value should match DAQ's

HTTP_EXPOSER_STARTED = False


def createExposer(prometheusCfgDir):
    logger = utils.get_logger(name="prometheus_manager.createExposer")
    if prometheusCfgDir == "":
        logger.warning(
            "Unable to update Prometheus configuration: directory not provided"
        )
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
            fileName = (
                f"{prometheusCfgDir}/drpmon_{hostname}_{port - PROM_PORT_BASE}.yaml"
            )
            if not os.path.exists(fileName):
                try:
                    with open(fileName, "wt") as f:
                        f.write(f"- targets:\n    - {hostname}:{port}\n")
                except Exception as ex:
                    logger.error(f"Error creating file {fileName}: {ex}")
                    return False
            else:
                pass  # File exists; no need to rewrite it
            logger.info(f"Providing run-time monitoring data on port {port}")
            HTTP_EXPOSER_STARTED = True
            return True
        except OSError:
            pass  # Port in use
        port += 1
    logger.error("No available port found for providing run-time monitoring")
    return False


def my_auth_handler(url, method, timeout, headers, data):
    # TODO Certificate and key files are hard-coded
    certfile = "/sdf/group/lcls/ds/ana/data/prom/promgwclient.pem"
    keyfile = "/sdf/group/lcls/ds/ana/data/prom/promgwclient.key"
    return tls_auth_handler(
        url,
        method,
        timeout,
        headers,
        data,
        certfile,
        keyfile,
        insecure_skip_verify=True,
    )


class PrometheusManager(object):
    registry = CollectorRegistry()
    # Store available psana metrics by metric_type, description, and labelnames
    metrics = {
        "psana_smd0_read": ("Gauge", "Disk reading rate (MB/s) for smd0", ()),
        "psana_smd0_rate": ("Gauge", "Processing rate by smd0 (kHz)", ()),
        "psana_smd0_wait": ("Gauge", "time spent (s) waiting for EventBuilders", ()),
        "psana_eb_rate": ("Gauge", "Processing rate by eb (kHz)", ()),
        "psana_eb_wait_smd0": ("Gauge", "time spent (s) waiting for Smd0", ()),
        "psana_eb_wait_bd": ("Gauge", "time spent (s) waiting for BigData cores", ()),
        "psana_bd_rate": ("Gauge", "Processing rate by bd (kHz)", ()),
        "psana_bd_read": ("Gauge", "Disk reading rate (MB/s) for bd", ()),
        "psana_bd_ana_rate": ("Gauge", "User-analysis rate on bd (kHz)", ()),
        "psana_bd_wait": ("Gauge", "time spent (s) waiting for EventBuilder", ()),
    }
    def __init__(self, job=None):
        self.username = getpass.getuser()
        self.rank = 0
        if job is None:
            default_job_id = os.environ.get("SLURM_JOB_ID", f"{self.username}")
            self.job = default_job_id
        else:
            self.job = job
        self.logger = utils.get_logger(name=utils.get_class_name(self))

    def set_rank(self, rank):
        self.rank = rank

    def register(self, metric_name):
        self.registry.register(metric_name)

    def push_metrics(self, e):
        # TODO: Certificate is read at every push, find a better way.
        while not e.isSet():
            if socket.gethostname().startswith("sdf"):
                push_to_gateway(
                    PUSH_GATEWAY,
                    job=self.job,
                    grouping_key={"rank": self.rank},
                    registry=self.registry,
                    handler=my_auth_handler,
                    timeout=None,
                )
            else:
                push_to_gateway(
                    PUSH_GATEWAY,
                    job=self.job,
                    grouping_key={"rank": self.rank},
                    registry=self.registry,
                    timeout=None,
                )
            self.logger.debug(
                f"TS: %s PUSHED JOBID:{self.job} RANK:{self.rank} {e.isSet()=}"
            )
            time.sleep(PUSH_INTERVAL_SECS)

    def delete_all_metrics_on_pushgateway(self, n_ranks=0):
        if not n_ranks:
            n_ranks = size
        for i_rank in range(n_ranks):
            args = [
                "curl",
                "-k",
                "-X",
                "DELETE",
                f"{PUSH_GATEWAY}/metrics/job/{self.job}/rank/{i_rank}",
            ]
            Popen(args)
            self.logger.debug(f"CLEANUP {args}")

    def create_exposer(self, prometheus_cfg_dir):
        return createExposer(prometheus_cfg_dir)

    def create_metric(self, metric_name):
        if metric_name in self.metrics:
            metric_type, desc, labelnames = self.metrics[metric_name]
            if metric_type == "Counter":
                self.registry.register(Counter(metric_name, desc, labelnames))
            elif metric_type == "Summary":
                self.registry.register(Summary(metric_name, desc, labelnames))
            elif metric_type == "Gauge":
                self.registry.register(Gauge(metric_name, desc, labelnames))
        else:
            self.logger.info(
                f"Warning: {metric_name} is not found in the list of available prometheus metrics"
            )

    def get_metric(self, metric_name):
        # get metric object from its name
        # NOTE that _created has to be appended to locate the key
        if metric_name in self.registry._names_to_collectors:
            collector = self.registry._names_to_collectors[metric_name]
        else:
            self.create_metric(metric_name)
            collector = self.registry._names_to_collectors[metric_name]
        return collector
