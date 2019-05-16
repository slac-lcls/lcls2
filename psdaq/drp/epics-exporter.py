from p4p.client.thread import Context
from prometheus_client import start_http_server
from prometheus_client.core import GaugeMetricFamily, REGISTRY

class CustomCollector():
    def __init__(self):
        self.pvactx = Context('pva')

    def collect(self):
        value = self.pvactx.get('DAQ:LAB2:PART:2:DeadFrac')
        print('collect', value.raw.value)
        yield GaugeMetricFamily('drp_dead_frac', value=value.raw.value)

REGISTRY.register(CustomCollector())

if __name__ == '__main__':
    # Start up the server to expose the metrics.
    c = CustomCollector()
    c.collect()
    start_http_server(9200)
    # Generate some requests.
    while True:
        pass
