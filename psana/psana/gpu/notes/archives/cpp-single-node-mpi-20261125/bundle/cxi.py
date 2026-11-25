"""Validated CXI sysfs counters, sampled around synchronized C++ read windows."""
import json
import os
from pathlib import Path
import re
import selectors
import subprocess
import time

COUNTERS = ['hni_sts_rx_ok_octets', 'hni_sts_tx_ok_octets', 'hni_sts_rx_octets_opt', 'hni_sts_tx_octets_opt', 'hni_sts_rx_octets_ieee', 'hni_sts_tx_octets_ieee']

def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def parse(raw):
    m = re.fullmatch(r'(\d+)@(\d+)\.(\d{1,9})\s*', raw)
    require(m is not None, 'unexpected CXI counter format: '+repr(raw))
    return {'value': int(m[1]), 'timestamp_ns': int(m[2])*10**9 + int(m[3].ljust(9, '0')), 'raw': raw.strip()}


def snapshot(devices):
    start = time.monotonic_ns()
    cxi = {}
    for device in devices:
        cxi[device.name] = {name: parse((device/'device/telemetry'/name).read_text()) for name in COUNTERS}
    net = {}
    for p in sorted(Path('/sys/class/net').glob('hsn*/statistics/*_bytes')):
        if re.fullmatch(r'hsn[0-9]+', p.parent.parent.name):
            net[str(p)] = int(p.read_text())
    finish = time.monotonic_ns()
    return {'monotonic_start_ns': start, 'monotonic_end_ns': finish, 'cxi': cxi, 'netdev': net}


def difference(a, b):
    require(a['cxi'].keys() == b['cxi'].keys(), 'NIC set changed')
    elapsed = ((b['monotonic_start_ns']+b['monotonic_end_ns']) - (a['monotonic_start_ns']+a['monotonic_end_ns'])) / 2e9
    require(elapsed > 0, 'nonpositive sample interval')
    per_nic = {}
    for nic in a['cxi']:
        per_nic[nic] = {}
        for name in COUNTERS:
            x,y = a['cxi'][nic][name], b['cxi'][nic][name]
            require(y['timestamp_ns'] > x['timestamp_ns'], 'counter timestamp did not advance')
            require(y['value'] >= x['value'], 'counter reset/wrap; cannot interpret')
            per_nic[nic][name] = y['value']-x['value']
    totals = {name: sum(v[name] for v in per_nic.values()) for name in COUNTERS}
    net = {k: b['netdev'][k]-v for k,v in a['netdev'].items() if k in b['netdev']}
    return {'seconds':elapsed,'per_nic':per_nic,'totals':totals,'netdev_delta':net,
            'rx_gib':totals[COUNTERS[0]]/2**30,'tx_gib':totals[COUNTERS[1]]/2**30,
            'rx_gib_s':totals[COUNTERS[0]]/2**30/elapsed,
            'netdev_rx_gib':sum(v for k,v in net.items() if k.endswith('/rx_bytes'))/2**30}


def audit(record, reader, required_nics=4):
    samples = record['samples']
    require(len(samples) >= 2, 'missing CXI boundary samples')
    expected = {'cxi%d' % i for i in range(required_nics)}
    require(set(samples[0]['cxi']) == expected, 'missing/unexpected CXI NICs')
    for a, b in zip(samples, samples[1:]):
        difference(a, b)  # Check every interval for reset/wrap, not just endpoints.
    summary = difference(samples[0], samples[-1])
    require(summary == record['summary'], 'CXI summary disagrees with raw samples')
    require(reader['cxi_handshake'] is True, 'reader did not use CXI gates')
    require(summary['seconds'] >= reader['seconds'], 'CXI interval excludes read time')
    require(reader['bytes'] > 0, 'empty rank read cannot be compared to CXI bytes')
    return {**summary, 'reader_bytes': reader['bytes'],
            'rx_over_read': summary['totals'][COUNTERS[0]] / reader['bytes'],
            'sampling_extra_seconds': summary['seconds'] - reader['seconds'],
            'sample_count': len(samples)}


def run_reader(command, stdout_path, stderr_path, *, env=None, required_nics=4,
               interval=0.25, timeout=900, snapshot_fn=snapshot, devices=None):
    """Collect in memory until DONE; preserve raw output and failures on disk.

    MPI processes keep launch descriptors. The C++ side has a barrier before
    READY and another after all GO acknowledgements. Final snapshots precede
    the global reductions. No auxiliary filesystem I/O occurs in the read loop.
    """
    require(interval > 0 and timeout > 0, 'invalid sampling interval/timeout')
    if devices is None:
        devices = sorted(Path('/sys/class/cxi').glob('cxi[0-9]*'))
    require(len(devices) == required_nics, 'expected %d CXI NICs' % required_nics)
    snapshot_fn(devices)  # Fail on missing permissions/counters before launching MPI child.
    child_env = dict(os.environ if env is None else env, CXI_HANDSHAKE='1')
    # Direct sysfs sampling replaces MPI stdout counter reports, keeping reader JSON distinct.
    child_env['MPICH_OFI_CXI_COUNTER_REPORT'] = '0'
    samples, lines = [], []
    failure = stdout_path.with_name(stdout_path.name + '.cxi-failure.json')
    record = {'interval_seconds': interval,
              'scope': 'single node, sampled once by rank zero; brackets all ranks after cache gates through last worker completion; includes barriers and sampling margin',
              'units': 'octets (bytes); hardware network traffic includes protocol/control/background bytes',
              'samples': samples}
    with stderr_path.open('x') as err:
        p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=err, text=True, bufsize=1, env=child_env, close_fds=False)
        selector = selectors.DefaultSelector()
        selector.register(p.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + timeout
        try:
            def read_line():
                remaining = deadline-time.monotonic()
                require(remaining > 0 and selector.select(remaining), 'CXI reader handshake timed out')
                line = p.stdout.readline()
                lines.append(line)
                return line.strip()
            require(read_line() == 'CXI_READY', 'reader failed before CXI_READY; inspect raw output')
            samples.append(snapshot_fn(devices))
            record['release_monotonic_ns'] = time.monotonic_ns()
            p.stdin.write('go\n'); p.stdin.flush()
            while True:
                remaining = deadline-time.monotonic()
                require(remaining > 0, 'reader exceeded timeout')
                if selector.select(min(interval, remaining)):
                    line = p.stdout.readline()
                    lines.append(line)
                    require(line.strip() == 'CXI_DONE', 'reader failed before CXI_DONE; inspect raw output')
                    break
                samples.append(snapshot_fn(devices))
            record['notice_monotonic_ns'] = time.monotonic_ns()
            samples.append(snapshot_fn(devices))
            p.stdin.write('done\n'); p.stdin.flush()
            stdout, _ = p.communicate(timeout=max(0.1, deadline-time.monotonic()))
            lines.append(stdout)
            require(p.returncode == 0, 'reader exited %s; inspect stderr' % p.returncode)
            reader = json.loads(stdout)
            require(reader.get('state') == 'complete', 'reader incomplete')
            record['summary'] = difference(samples[0], samples[-1])
            record['comparison'] = audit(record, dict(reader, bytes=reader['global_bytes'], seconds=reader['global_seconds']), required_nics)
            return reader, record
        except BaseException as exc:
            record['error'] = repr(exc)
            save(failure, record)
            raise
        finally:
            selector.close()
            if p.poll() is None:
                p.kill()
                p.wait()
            try:
                p.stdin.close()
            except BrokenPipeError:
                pass
            p.stdout.close()
            stdout_path.write_text(''.join(lines))
