"""Small read-only snapshots bracketing the collective event loop."""
from pathlib import Path
import time

def capture():
    result={'monotonic':time.monotonic(),'time':time.time(),'files':{}}
    paths=[Path('/proc/diskstats'),Path('/proc/net/dev')]
    for name in ('hni_sts_rx_ok_octets','hni_sts_tx_ok_octets'):
        paths.extend(Path('/sys/class/cxi').glob('*/device/telemetry/'+name))
    for path in paths:
        try:result['files'][str(path)]=path.read_text()
        except OSError as e:result['files'][str(path)]=str(e)
    return result
