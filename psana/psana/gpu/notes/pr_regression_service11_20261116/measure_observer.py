"""Local observer microbenchmark; no psana import, dataset, MPI, or CUDA.

Run with Python 3.11 from any directory in this checkout. Compare the baseline
observer with the updated observer using identical service-12 stub envelopes.
This measures the observer wrapper only, not native end-to-end overhead.
"""
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[5]
BASE = '980a7e7e3611c7561e4647af7d2fa0e126da8b94'
WORKER = 'psana/psana/gpu/scripts/performance/regression/native_worker.py'
sys.path.insert(0, str((ROOT / WORKER).parent))
import native_worker
spec = importlib.util.spec_from_file_location('transitionid', ROOT / 'psana/psana/psexp/transitionid.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
old = {}
exec(compile(subprocess.check_output(['git', '-C', str(ROOT), 'show', BASE + ':' + WORKER]), '<baseline-worker>', 'exec'), old)

N = 100000
REPEATS = 9

def sample(observer, updated):
    envelope = SimpleNamespace(service=12)
    class Events:
        def __init__(self):
            self._batch_source = None
            self._batch_event_count = 0
            self._batch_start_time = 1
        def __next__(self):
            self._batch_event_count += 1
            return envelope
        def _emit_batch_end(self):
            self._batch_event_count = 0
            self._batch_start_time = None
    args = (Events, lambda: 1.0, lambda e: e.service)
    observer = observer(*args, module.TransitionId.isEvent) if updated else observer(*args)
    try:
        events = Events()
        start = time.perf_counter_ns()
        for _ in range(N):
            next(events)
        elapsed = time.perf_counter_ns() - start
        events._emit_batch_end()
        assert observer.result(N)['l1_per_delivery_window'] == [N]
        return elapsed / N
    finally:
        observer.restore()

variants = {'baseline': (old['DeliveryAccounting'], False),
            'isEvent': (native_worker.DeliveryAccounting, True)}
for args in variants.values():
    sample(*args)
rows = []
for repeat in range(REPEATS):
    order = list(variants) if repeat % 2 == 0 else list(reversed(variants))
    rows.append({name: sample(*variants[name]) for name in order})
medians = {name: statistics.median(row[name] for row in rows) for name in variants}
deltas = [row['isEvent'] - row['baseline'] for row in rows]
print(json.dumps(dict(python=sys.version, host=platform.node(), base=BASE,
                     samples=rows, events_per_sample=N, repeats=REPEATS,
                     median_ns_per_event=medians,
                     median_paired_delta_ns_per_event=statistics.median(deltas),
                     limitation='Login-node synthetic service-12 observer only; not native throughput or full instrumentation qualification.'), indent=2))
