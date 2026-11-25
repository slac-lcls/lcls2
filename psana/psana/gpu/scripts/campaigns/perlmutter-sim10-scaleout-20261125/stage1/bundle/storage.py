"""Owner-scoped count2 Lustre cache evidence; no remote-traffic claim."""
from pathlib import Path
import time
import math
from qualified_support import cache as scan, diagnostics
from support import input_info, require, write

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


CACHE_MAX_ATTEMPTS = 3
CACHE_SECONDS = 180
CACHE_RETRY_PAUSE = 1.0
CACHE_VERIFY_PAUSE = 0.1


def validate_cache_gate(receipt, files):
    """Audit successful conditioning without trusting just the final zero scan."""
    require(receipt['status'] == 'pass', 'Cache gate status')
    require(receipt['max_attempts'] == CACHE_MAX_ATTEMPTS and
            receipt['budget_seconds'] == CACHE_SECONDS, 'Cache gate policy')
    require(math.isfinite(receipt['elapsed_seconds']) and
            0 <= receipt['elapsed_seconds'] < CACHE_SECONDS, 'Cache conditioning deadline')
    attempts = receipt['attempts']
    require(1 <= len(attempts) <= CACHE_MAX_ATTEMPTS, 'Cache attempt limit')
    expected_pages = None
    previous = 0.0
    for index, attempt in enumerate(attempts, 1):
        require(attempt['number'] == index, 'Cache attempt sequence')
        require([s['phase'] for s in attempt['scans']] == ['evict', 'verify'], 'Two cache scans')
        cold = []
        for scan_result in attempt['scans']:
            elapsed = scan_result['elapsed_seconds']
            require(math.isfinite(elapsed) and previous <= elapsed <= receipt['elapsed_seconds'], 'Cache scan timing')
            if scan_result['phase'] == 'verify':
                require(elapsed - previous >= CACHE_VERIFY_PAUSE - 1e-6, 'Cache verification spacing')
            elif index > 1:
                require(elapsed - previous >= CACHE_RETRY_PAUSE - 1e-6, 'Cache retry spacing')
            previous = elapsed
            rows = scan_result['files']
            require([r['path'] for r in rows] == files and len(set(files)) == len(files) > 0, 'Cache file coverage')
            require(all(type(r['pages']) is int and type(r['resident_pages']) is int and
                        0 <= r['resident_pages'] <= r['pages'] and r['pages'] > 0 and
                        r['resident'] == r['resident_pages'] for r in rows), 'Cache page counts')
            pages = [r['pages'] for r in rows]
            if expected_pages is None: expected_pages = pages
            require(pages == expected_pages, 'Cache file sizes changed')
            cold.append(all(r['resident_pages'] == 0 for r in rows))
        require(all(cold) == (index == len(attempts)), 'Only final cache attempt passes')
    return attempts[-1]['scans']


def condition_cache(files, directory, media, receipt_path):
    """Three bounded attempts; persist each scan before enforcing the cold gate.

    The monotonic deadline is checked around synchronous scans. A blocked syscall
    cannot be interrupted here; the enclosing srun/process timeout is the hard cap.
    """
    started = time.monotonic()
    receipt = dict(status='running', max_attempts=CACHE_MAX_ATTEMPTS,
                   budget_seconds=CACHE_SECONDS, elapsed_seconds=0.0, attempts=[])

    def save():
        receipt['elapsed_seconds'] = time.monotonic() - started
        write(receipt_path, receipt)

    def check_deadline():
        require(time.monotonic() - started < CACHE_SECONDS, 'Cache conditioning deadline')

    save()
    try:
        for number in range(1, CACHE_MAX_ATTEMPTS + 1):
            check_deadline()
            if number > 1:
                time.sleep(CACHE_RETRY_PAUSE)
                check_deadline()
            attempt = dict(number=number, scans=[])
            receipt['attempts'].append(attempt)
            save()
            for phase in ('evict', 'verify'):
                if phase == 'verify': time.sleep(CACHE_VERIFY_PAUSE)
                check_deadline()
                rows = cache(files, directory, media, evict=(phase == 'evict'))
                attempt['scans'].append(dict(phase=phase, files=rows,
                                            elapsed_seconds=time.monotonic() - started))
                save()
                check_deadline()
            if all(row['resident_pages'] == 0 for scan_result in attempt['scans']
                   for row in scan_result['files']):
                receipt['status'] = 'pass'
                save()
                validate_cache_gate(receipt, files)
                check_deadline()
                return receipt
        raise ValueError('Cold host-page-cache gate: three attempts exhausted')
    except Exception as error:
        receipt.update(status='failed', error=repr(error))
        save()
        raise
