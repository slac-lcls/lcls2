import time
from prometheus_client import CollectorRegistry, Counter, push_to_gateway
import logging

PUSH_INTERVAL_SECS  = 5
PUSH_GATEWAY        = 'psdm03:9091'

registry = CollectorRegistry()
metrics = {'psana_smd0_read'   : 'events read by Smd0', 
        'psana_smd0_sent'      : 'events sent by Smd0', 
        'psana_eb_sent'        : 'events sent by EventBuilder',
        'psana_bd_read'        : 'events read by BigData', 
        }

for metric_name, desc in metrics.items():
    try:
        registry.register(Counter(metric_name, desc, ['unit', 'endpoint']))
    except:
        logging.debug('metric_name: %s registry: %s'%(metric_name, registry._names_to_collectors))
        exit()

class PrometheusManager(object):
    def __init__(self):
        pass

    def push_metrics(self, e, from_whom=''):
        while not e.isSet():
            push_to_gateway(PUSH_GATEWAY, job='psana_pushgateway', grouping_key={'pid': from_whom}, registry=registry)
            logging.debug('pushed from pid %s %s e.isSet()=%s'%(from_whom, time.time(), e.isSet()))
            time.sleep(PUSH_INTERVAL_SECS)
        
    def get_counter(self, metric_name):
        # get metric object from its name 
        # NOTE that _created has to be appended to locate the key
        c = registry._names_to_collectors[f'{metric_name}_created']
        return c
        

