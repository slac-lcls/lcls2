"""Read-only Lustre/client/node and cgroup observations; no process attribution."""
import glob
import json
from pathlib import Path
import threading
import time


def inventory():
    patterns = ['/proc/fs/lustre/llite/*/stats', '/sys/fs/lustre/llite/*/stats',
                '/sys/class/net/*/statistics/rx_bytes', '/sys/class/net/*/statistics/tx_bytes',
                '/sys/class/cxi/*/device/telemetry/*']
    matches = {pattern: [p for p in glob.glob(pattern) if Path(p).is_file()] for pattern in patterns}
    paths = sorted({p for group in matches.values() for p in group})
    # Resolve the actual batch process cgroup, never assume the root is the job.
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        hierarchy, controllers, relative = line.split(':', 2)
        if not controllers:
            base = Path('/sys/fs/cgroup') / relative.lstrip('/')
            # Include job ancestors: srun ranks live in a sibling step to the batch shell.
            for parent in [base] + list(base.parents):
                if parent == Path('/sys/fs/cgroup') or Path('/sys/fs/cgroup') in parent.parents:
                    paths += [str(parent/name) for name in ['memory.current','memory.max','memory.events','memory.stat','memory.pressure']]
        elif 'memory' in controllers.split(','):
            base = Path('/sys/fs/cgroup/memory') / relative.lstrip('/')
            paths += [str(base/name) for name in ['memory.usage_in_bytes','memory.limit_in_bytes','memory.stat','memory.failcnt']]
    paths += ['/proc/meminfo', '/proc/vmstat', '/proc/pressure/memory', '/proc/self/cgroup']
    return dict(paths=sorted(set(paths)), pattern_counts={k:len(v) for k,v in matches.items()},
                unavailable=[k for k,v in matches.items() if not v],
                lustre_stats_observable=any(v for k,v in matches.items() if 'llite' in k), scope='node/client counters and batch cgroup; not process-attributed storage traffic',
                limitation='Cache uncontrolled; network traffic need not be Lustre; unavailable counters recorded explicitly.')


def snapshot(network):
    values, errors = {}, {}
    for name in network['paths']:
        try:
            values[name] = Path(name).read_text()
        except (OSError, UnicodeError) as exc:
            errors[name] = str(exc)
    return dict(unix_time=time.time(), monotonic=time.monotonic(), values=values, errors=errors,
                scope=network['scope'])


def delta(before, after):
    return dict(before=before, after=after, seconds=after['monotonic']-before['monotonic'],
                scope=after['scope'])


class Monitor:
    def __init__(self, network, output):
        self.network, self.output = network, Path(output)
        self.stop_event = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True)
    def run(self):
        try:
            with self.output.open('x') as stream:
                while not self.stop_event.is_set():
                    stream.write(json.dumps(snapshot(self.network))+'\n'); stream.flush()
                    self.stop_event.wait(2)
        except BaseException as exc:
            self.error = repr(exc)
    def start(self):
        self.thread.start()
    def stop(self):
        self.stop_event.set(); self.thread.join(10)
        if self.thread.is_alive() or self.error:
            raise RuntimeError(f'Storage monitor failure: {self.error}')
