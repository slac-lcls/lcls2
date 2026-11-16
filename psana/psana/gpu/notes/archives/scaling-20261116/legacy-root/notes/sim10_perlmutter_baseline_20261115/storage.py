"""Owner-scoped count2 Lustre cache evidence; no remote-traffic claim."""
from pathlib import Path
import time
from qualified_support import cache as scan, diagnostics
from support import input_info, require

def cache(files, directory, media, evict=False):
    info = input_info()
    require(media == 'lustre' and str(directory) == info['directory'] and files == info['files'],
            'Cache operation outside qualified count2 allowlist')
    return [dict(row, resident_pages=row['resident']) for row in scan(info['payload_records'], evict)]

def counters(leaves):
    return dict(time=time.time(), evidence=diagnostics(), remote_traffic_verified=False)

def evidence(observation, media, logical_bytes, seconds):
    require(media == 'lustre' and seconds > 0 and logical_bytes > 0, 'Storage evidence arguments')
    require(all(row['resident_pages'] == 0 for row in observation['initial_cache']), 'Local cold gate')
    return dict(local_page_cache_cold=True, remote_traffic_verified=False,
                logical_payload_bytes=logical_bytes, logical_payload_gib_s=logical_bytes/2**30/seconds,
                interpretation='Logical payload; client residency only. Backend caches and wire traffic unverified.')
