import time
from p4p.client.thread import Context
from prometheus_client import start_http_server
from prometheus_client.core import GaugeMetricFamily, REGISTRY

class CustomCollector():
    def __init__(self):
        self.pvactx = Context('pva')

    def collect(self):
        g = GaugeMetricFamily('drp_dead_frac', documentation='',  labels=['partition'])
        for p in range(8):
            value = self.pvactx.get('DAQ:LAB2:XPM:2:PART:%d:DeadFrac' %p)
            print('collect', value.raw.value)
            g.add_metric([str(p)], value.raw.value)
        yield g

REGISTRY.register(CustomCollector())

if __name__ == '__main__':
    # Start up the server to expose the metrics.
    c = CustomCollector()
    c.collect()
    start_http_server(9200)
    while True:
        time.sleep(5)
