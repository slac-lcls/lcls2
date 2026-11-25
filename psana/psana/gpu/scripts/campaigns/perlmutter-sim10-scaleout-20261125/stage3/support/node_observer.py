"""Node-scoped memory and interface evidence; no claim of exclusive NIC traffic."""
from pathlib import Path
import time
from memory_watch import Watch,validate_memory


def network():
    result={}
    for p in Path('/sys/class/net').iterdir():
        if p.name=='lo':continue
        try:result[p.name]={k:int((p/'statistics'/k).read_text()) for k in ('rx_bytes','tx_bytes')}
        except (OSError,ValueError):continue
    return dict(monotonic=time.monotonic(),interfaces=result,
                scope='Host interface counters; includes unrelated traffic, not storage attribution')


def start_memory(uuids):
    watches=[Watch(uuid) for uuid in uuids]
    for w in watches:w.start()
    return watches


def stop_memory(watches):
    rows=[w.stop() for w in watches]
    for row in rows:validate_memory(row)
    return rows
