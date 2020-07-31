import time
from prometheus_client import CollectorRegistry, Counter, push_to_gateway, Summary
import logging

PUSH_INTERVAL_SECS  = 5
PUSH_GATEWAY        = 'psdm03:9091'

registry = CollectorRegistry()
metrics ={'psana_smd0_wait_disk': ('Summary', 'Time spent (s) reading smalldata'),
        'psana_smd0_read'       : ('Counter', 'Counting no. of events/batches/MB read by Smd0'), 
        'psana_smd0_sent'       : ('Counter', 'Counting no. of events/batches/MB and wait time \
                                    communicating with EventBuilder cores'), 
        'psana_eb_sent'         : ('Counter', 'Counting no. of events/batches/MB and wait time \
                                    communicating with BigData cores'),
        'psana_eb_filter'       : ('Counter', 'Counting no. of batches and wait time \
                                    in filter callback'), 
        'psana_bd_read'         : ('Counter', 'Counting no. of events processed by BigData'),
        'psana_bd_wait_disk'    : ('Summary', 'time spent (s) reading bigdata'),
        'psana_bd_ana'          : ('Counter', 'time spent (s) in analysis fn on \
                                    BigData core'),
        }

for metric_name, (metric_type, desc) in metrics.items():
    if metric_type == 'Counter':
        registry.register(Counter(metric_name, desc, ['unit', 'endpoint']))
    elif metric_type == 'Summary':
        registry.register(Summary(metric_name, desc))

class PrometheusManager(object):
    def __init__(self, jobid):
        self.jobid = jobid
        pass

    def push_metrics(self, e, from_whom=''):
        while not e.isSet():
            push_to_gateway(PUSH_GATEWAY, job='psana_pushgateway', grouping_key={'jobid': self.jobid, 'rank': from_whom}, registry=registry)
            logging.debug('TS: %s PUSHED JOBID: %s RANK: %s e.isSet():%s'%(time.time(), self.jobid, from_whom, e.isSet()))
            time.sleep(PUSH_INTERVAL_SECS)
        
    @staticmethod
    def get_metric(metric_name):
        # get metric object from its name 
        # NOTE that _created has to be appended to locate the key
        collector = registry._names_to_collectors[f'{metric_name}_created']
        return collector
        
        

