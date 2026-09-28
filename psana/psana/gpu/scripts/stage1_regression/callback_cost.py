"""Internal producer scheduling costs; no disk I/O or automatic output D2H.

Run the same work against frozen per-event and batched runtimes. Parsing,
compilation, fixture uploads, preallocation and correctness copies are outside
timing. The Jungfrau-shaped case is synthetic, not end-to-end JF throughput.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace as NS

import numpy as np


@contextmanager
def count_framework(cp, gd, parser_module, dispatch):
    """Count hot-path operations separately; restore every hook before timing.

    Gather-map copies are logical upload calls (one .set in prepare), not a
    CUDA API trace. Task metadata .set and completion-event creation are hooked.
    No fixture setup or diagnostic D2H is included in these counters.
    """
    counts = dict(walk=0, init=0, locate=0, gather=0, completion_events=0,
                  gather_map_uploads=0, task_metadata_uploads=0)
    originals = []

    def replace(module, name, value):
        originals.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    for module, name, label in [(parser_module, '_walk_kernel', 'walk'),
            (parser_module, '_init_locators_kernel', 'init'),
            (parser_module, '_locate_fields_kernel', 'locate'),
            (gd, '_batched_gather_kernel', 'gather')]:
        original = getattr(module, name)
        def factory(*args, _original=original, _label=label):
            kernel = _original(*args)
            def launch(*args, **kwargs):
                counts[_label] += 1
                return kernel(*args, **kwargs)
            return launch
        replace(module, name, factory)
    event = cp.cuda.Event
    def make_event(*args, **kwargs):
        counts['completion_events'] += 1
        return event(*args, **kwargs)
    replace(cp.cuda, 'Event', make_event)
    prepare = gd._GatherMap.prepare
    def prepare_map(*args, **kwargs):
        value = prepare(*args, **kwargs)
        counts['gather_map_uploads'] += 1
        return value
    replace(gd._GatherMap, 'prepare', prepare_map)
    if dispatch == 'batch':
        from psana.gpu import gpu_task_batch as tb
        allocate = tb.owned_empty
        class Upload:
            def __init__(self, array): self.array = array
            def __getattr__(self, name): return getattr(self.array, name)
            def __getitem__(self, key): return self.array[key]
            def set(self, source, **kwargs):
                counts['task_metadata_uploads'] += 1
                return self.array.set(source, **kwargs)
        replace(tb, 'owned_empty', lambda *args, **kwargs: Upload(allocate(*args, **kwargs)))
    try:
        yield counts
    finally:
        for module, name, original in reversed(originals):
            setattr(module, name, original)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dispatch', choices=('event', 'batch'), default='batch')
    p.add_argument('--profile', choices=('micro', 'jungfrau'), default='micro')
    p.add_argument('--repetitions', type=int, default=6)
    p.add_argument('--submissions', type=int, default=200)
    p.add_argument('--reverse', action='store_true')
    a = p.parse_args()
    if a.repetitions < 1 or a.submissions < 1:
        p.error('use positive submissions and repetitions')
    import cupy as cp
    import psana
    from psana.gpu import GpuTask, gpu_detector as gd
    from psana.gpu.gpu_budget import _GpuBudget
    from psana.gpu.gpu_input_window import InputWindow
    from psana.gpu.gpu_stream import EventPool
    from psana.gpu.gpudgram import parser as parser_module
    sys.path.insert(0, str(Path(psana.__file__).parent/'tests/gpu/integration'))
    from test_batched_gather import _setup, _input
    from callback_fixture import jungfrau_fixture

    scratch_kernel = cp.RawKernel('''extern "C" __global__ void scratch(
        const unsigned short* raw, unsigned short* out, unsigned long long n) {
        unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(i<n) out[i]=raw[i]+1;
    }''', 'scratch')
    publish_kernel = cp.RawKernel('''extern "C" __global__ void publish(
        const unsigned short* raw, unsigned int* out, unsigned long long n,
        unsigned long long stride) {
        unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(i<n) out[i]=raw[i*stride+300]+1;
    }''', 'publish')
    scratch_kernel.compile()
    publish_kernel.compile()
    modes = ('none', 'empty', 'scratch', 'publish', 'scratch_prealloc', 'publish_prealloc')
    result = dict(dispatch=a.dispatch, profile=a.profile, scope=__doc__, psana=psana.__file__,
                  cupy=cp.__version__, cuda=cp.cuda.runtime.runtimeGetVersion(),
                  device=str(cp.cuda.runtime.getDeviceProperties(0)['name']),
                  repetitions=a.repetitions, submissions=a.submissions, preflights=[], samples=[])
    a.output.parent.mkdir(parents=True, exist_ok=True)

    for batch_size in (1, 3, 20):
        budget = _GpuBudget(8*1024**3)
        if a.profile == 'jungfrau':
            parser, initial, dtype, batch, events, expected = jungfrau_fixture(cp, budget, batch_size)
        else:
            parser, initial, dtype = _setup(cp, budget=budget)
            producer = cp.cuda.Stream(non_blocking=True)
            batch, events, expected = _input(cp, parser, producer, [(0,)]*batch_size, dtype)
            producer.synchronize()
            expected = np.asarray(expected)
        specs = tuple(e.event for e in events)
        indices = {e.timestamp: i for i, e in enumerate(specs)}
        descriptors = batch._test_descriptors
        envelopes = [NS(dgrams=[NS(timestamp=lambda ts=e.timestamp: ts)]) for e in specs]
        gv = NS(iter_events=lambda: iter(specs))
        stride = int(np.prod(initial.det_shape))

        class ImmutableParsedInput:
            # Parsed raw/locator arrays are immutable and retained through drains.
            def __getattr__(self, name): return getattr(batch, name)
            def retire(self): pass

        for depth in (1, 2):
            preparer = gd.DenseInputPreparer(initial.det_shape, initial.binding,
                                            n_slots=depth, budget=budget)
            preparer.configure_gather(parser.handle_indices)
            cp.cuda.get_current_stream().synchronize()

            def run_case(mode, submissions, *, check=False):
                pool = EventPool(n=depth, **({'budget': budget} if a.dispatch == 'batch' else {}))
                counts = dict(callbacks=0, user_allocations=0, user_launches=0,
                              checked_submissions=0, diagnostic_d2h_copies=0)
                is_scratch = mode.startswith('scratch')
                is_publish = mode.startswith('publish')
                preallocated = mode.endswith('_prealloc')
                # One allocation per slot, with row views built outside timing.
                buffers, views = [], []
                if preallocated:
                    for _ in range(depth):
                        buf = cp.empty((batch_size, *initial.det_shape) if is_scratch else (batch_size,),
                                       cp.uint16 if is_scratch else cp.uint32)
                        buffers.append(buf)
                        views.append([buf[i] if is_scratch else buf[i:i+1].reshape(())
                                      for i in range(batch_size)])
                cp.cuda.get_current_stream().synchronize()

                def callback(ctx, stream):
                    if check: counts['callbacks'] += 1
                    if mode == 'empty': return
                    raw = ctx.input('camera.raw')
                    n = batch_size if a.dispatch == 'batch' else 1
                    if preallocated:
                        slot = pool.next_slot_id
                        out = buffers[slot] if a.dispatch == 'batch' else views[slot][indices[ctx.timestamp]]
                    else:
                        out = cp.empty_like(raw) if is_scratch else cp.empty(
                            (n,) if a.dispatch == 'batch' else (), cp.uint32)
                        if check: counts['user_allocations'] += 1
                    if is_scratch:
                        ctx.keepalive(out)
                        scratch_kernel(((raw.size+255)//256,), (256,),
                                       (raw, out, np.uint64(raw.size)), stream=stream)
                    else:
                        ctx.publish('value', out)
                        publish_kernel(((n+255)//256,), (256,),
                                       (raw, out, np.uint64(n), np.uint64(stride)), stream=stream)
                    if check: counts['user_launches'] += 1

                def validate(record):
                    counts['checked_submissions'] += 1
                    # Retirement or flush already synchronized this producer.
                    np.testing.assert_array_equal(record.prepared_inputs['camera.raw'].data.get(), expected)
                    counts['diagnostic_d2h_copies'] += 1
                    if is_publish:
                        assert len(record.publications_by_ts) == batch_size
                        for i, event in enumerate(specs):
                            value = record.publications_by_ts[event.timestamp]['value']
                            assert value.shape == () and value.nbytes == 4
                            assert int(cp.asnumpy(value.array)) == int(expected[i].reshape(-1)[300])+1
                            counts['diagnostic_d2h_copies'] += 1
                    else:
                        assert not record.publications_by_ts
                    if is_scratch:
                        arrays = [x for x in record.producer_owners if isinstance(x, cp.ndarray)]
                        assert len(arrays) == (1 if a.dispatch == 'batch' else batch_size)
                        for i, array in enumerate(arrays):
                            reference = expected if a.dispatch == 'batch' else expected[i]
                            np.testing.assert_array_equal(array.get(), reference + np.uint16(1))
                            counts['diagnostic_d2h_copies'] += 1

                task = None if mode == 'none' else GpuTask(callback, ['camera.raw'])
                options = dict(task=task, detector_bindings={'camera': initial.binding})
                submit_ns = retire_ns = 0
                total_start = time.perf_counter_ns()
                for _ in range(submissions):
                    t = time.perf_counter_ns()
                    old = pool.begin_retire_next()
                    if check and old is not None: validate(old)
                    pool.finish_retire_next()
                    retire_ns += time.perf_counter_ns()-t
                    window = InputWindow(7, 0, ImmutableParsedInput(), descriptors)
                    try:
                        t = time.perf_counter_ns()
                        pool.submit(gv, None, envelopes, {'camera.raw': preparer},
                                    input_windows=(window,), batch_id=7, **options)
                        submit_ns += time.perf_counter_ns()-t
                    finally:
                        window.close()
                t = time.perf_counter_ns()
                for record in pool.flush():
                    if check: validate(record)
                retire_ns += time.perf_counter_ns()-t
                total_ns = time.perf_counter_ns()-total_start
                assert not pool.active_count and batch._locators == {}
                if check:
                    calls = 0 if mode == 'none' else submissions*(1 if a.dispatch == 'batch' else batch_size)
                    assert counts['callbacks'] == calls, counts
                    assert counts['user_launches'] == (calls if is_scratch or is_publish else 0), counts
                    assert counts['user_allocations'] == (calls if (is_scratch or is_publish) and not preallocated else 0), counts
                    assert counts['checked_submissions'] == submissions
                output_bytes = batch_size*(stride*2 if is_scratch else 4 if is_publish else 0)
                row = dict(mode=mode, batch_size=batch_size, depth=depth, submissions=submissions,
                           events=batch_size*submissions, submit_ns=submit_ns,
                           retire_ns=retire_ns, loop_ns=total_ns,
                           user_output_bytes_per_subbatch=output_bytes,
                           user_output_live_bytes_bound=output_bytes*(depth if preallocated else min(depth, submissions)),
                           preallocated_buffers=len(buffers),
                           budget_committed_after_drain=budget.committed(),
                           cupy_pool_used_after_drain=cp.get_default_memory_pool().used_bytes())
                for name, ns in [('submit', submit_ns), ('retire', retire_ns), ('loop', total_ns)]:
                    row[name+'_us_per_subbatch'] = ns/submissions/1000
                    row[name+'_us_per_event'] = ns/(batch_size*submissions)/1000
                return row, counts

            for mode in modes:
                run_case(mode, 5)
                ncheck = depth*2+1  # Reuse every slot, and validate every output.
                with count_framework(cp, gd, parser_module, a.dispatch) as framework:
                    _, counts = run_case(mode, ncheck, check=True)
                assert framework == dict(walk=0, init=0, locate=0, gather=ncheck,
                    completion_events=ncheck, gather_map_uploads=ncheck,
                    task_metadata_uploads=ncheck if a.dispatch == 'batch' and mode != 'none' else 0), framework
                result['preflights'].append(dict(mode=mode, batch_size=batch_size, depth=depth,
                    submissions=ncheck, framework=framework, user=counts))
            for rep in range(1, a.repetitions+1):
                order = modes if (rep % 2 == 1) != a.reverse else tuple(reversed(modes))
                for mode in order:
                    row, _ = run_case(mode, a.submissions)
                    row['repetition'] = rep
                    result['samples'].append(row)
                    a.output.write_text(json.dumps(result, indent=2)+'\n')
            preparer.trim_slot_buffers()
        batch.retire()
    result['summary'] = []
    for batch_size in (1, 3, 20):
        for depth in (1, 2):
            for mode in modes:
                rows = [r for r in result['samples'] if (r['batch_size'], r['depth'], r['mode']) == (batch_size, depth, mode)]
                summary = dict(batch_size=batch_size, depth=depth, mode=mode)
                for name in ('submit', 'retire', 'loop'):
                    for unit in ('subbatch', 'event'):
                        key = name+'_us_per_'+unit
                        summary['median_'+key] = statistics.median(r[key] for r in rows)
                result['summary'].append(summary)
    result['complete'] = True
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print('CALLBACK_COST_COMPLETE', a.dispatch, a.profile, flush=True)


if __name__ == '__main__':
    main()
