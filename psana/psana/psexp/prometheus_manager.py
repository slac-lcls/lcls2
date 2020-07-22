import time
from prometheus_client import CollectorRegistry, Counter, push_to_gateway, Summary
import logging

PUSH_INTERVAL_SECS  = 5
PUSH_GATEWAY        = 'psdm03:9091'

registry = CollectorRegistry()
metrics = {'psana_smd0_read'   : ('Counter', 'events read by Smd0'), 
        'psana_smd0_sent'      : ('Counter', 'events sent by Smd0'), 
        'psana_eb_sent'        : ('Counter', 'events sent by EventBuilder'),
        'psana_bd_read'        : ('Counter', 'events read by BigData'),
        'psana_smd0_wait_disk' : ('Summary', 'time spend reading smalldata'),
        }

for metric_name, (metric_type, desc) in metrics.items():
    if metric_type == 'Counter':
        registry.register(Counter(metric_name, desc, ['unit', 'endpoint']))
    elif metric_type == 'Summary':
        registry.register(Summary(metric_name, desc))

class PrometheusManager(object):
    def __init__(self):
        pass

    def push_metrics(self, e, from_whom=''):
        while not e.isSet():
            push_to_gateway(PUSH_GATEWAY, job='psana_pushgateway', grouping_key={'pid': from_whom}, registry=registry)
            logging.debug('pushed from pid %s %s e.isSet()=%s'%(from_whom, time.time(), e.isSet()))
            time.sleep(PUSH_INTERVAL_SECS)
        
    def get_metric(self, metric_name):
        # get metric object from its name 
        # NOTE that _created has to be appended to locate the key
        collector = registry._names_to_collectors[f'{metric_name}_created']
        return collector
        
        

