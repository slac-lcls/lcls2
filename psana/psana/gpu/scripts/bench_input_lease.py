"""Host event/lease overhead, without I/O, CUDA submission or field copies.

Run in fresh baseline/candidate processes on the same CPU affinity. This isolates
Python lifetime bookkeeping; it is not a detector-throughput measurement.
"""
import argparse
import json
import time
from types import SimpleNamespace as NS

import numpy as np

from psana.gpu.context import GpuEventState
from psana.gpu.gpu_input import GpuEventDgrams, InputSlotLease
from psana.gpu.gpu_input_window import InputWindow
from psana.gpu.gpu_kvikio_read import DESC_NCOLS


def measure(events, views):
    class Token:
        waits = 0

        def synchronize(self):
            self.waits += 1

    ready, done = Token(), Token()
    batch = NS(n_dgrams=0, stream_ids_by_dgram=np.empty(0, np.int32), walk_done=ready)
    owner = InputWindow(0, 0, batch, np.empty((0, DESC_NCOLS), np.uint64))
    event = NS(first_desc=0, n_desc=0, timestamp=1, batch_event_index=0)
    started = time.perf_counter()
    for _ in range(events):
        parent = InputSlotLease(ready, (owner,))
        dgrams = GpuEventDgrams(event, batch)
        dgrams.bind_lease(parent)
        state = GpuEventState({}, event_dgrams=dgrams, input_lease=parent)
        if views:
            child = parent.acquire_view()
            child.register_consumer_done(done)
        parent.wait_until_safe_to_reuse()
        if views:
            child.wait_until_safe_to_reuse()
    seconds = time.perf_counter() - started
    assert state._input_lease is parent
    assert owner.references == 0 and ready.waits == done.waits == 0
    owner.close()
    assert ready.waits == 1 and done.waits == int(views)
    return dict(events=events, loop_s=seconds, events_per_s=events/seconds,
                host_us_per_event=seconds*1e6/events, child_view=views)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=int, default=20000)
    parser.add_argument('--repeat', type=int, default=5)
    args = parser.parse_args()
    assert args.events > 0 and args.repeat > 0
    for views in (False, True):
        measure(2000, views)
        for _ in range(args.repeat):
            print(json.dumps(measure(args.events, views)), flush=True)


if __name__ == '__main__':
    main()
