from .datasource import DataSource
#from .smalldata import SmallData
from psana.psexp.prometheus_manager import PrometheusManager
import time
g_ts = PrometheusManager.get_metric('psana_timestamp')
g_ts.labels('psana_init').set(time.time())
