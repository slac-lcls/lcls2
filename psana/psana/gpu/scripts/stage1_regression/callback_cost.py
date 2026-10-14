"""Synthetic Stage 3 producer costs; no disk I/O or automatic output D2H.

Reuse the real-device acceptance fixture's immutable parsed uint16 payload.
Fresh input-window facades exercise normal execution leases without accumulating
resident-window dependencies across repetitions. Parsing/compilation/setup are
outside timing. The 900-pixel synthetic input is not a Jungfrau throughput test.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace as NS

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repetitions', type=int, default=6)
    p.add_argument('--submissions', type=int, default=200)
    a = p.parse_args()
    if a.repetitions < 2 or a.repetitions % 2 or a.submissions < 1:
        p.error('use positive submissions and an even repetition count >= 2')
    import cupy as cp
    import psana
    from psana.gpu import GpuTask, gpu_detector as gd
    from psana.gpu.gpu_budget import _GpuBudget
    from psana.gpu.gpu_input_window import InputWindow
    from psana.gpu.gpu_stream import EventPool
    from psana.gpu.gpudgram import parser as parser_module
    sys.path.insert(0, str(Path(psana.__file__).parent/'tests/gpu/integration'))
    from test_batched_gather import _setup, _input

    scratch_kernel = cp.RawKernel('''extern "C" __global__ void scratch(
        const unsigned short* raw, unsigned short* out) {
        unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
        if(i<900) out[i]=raw[i]+1;
    }''', 'scratch')
    publish_kernel = cp.RawKernel('''extern "C" __global__ void publish(
        const unsigned short* raw, unsigned int* out) { *out=raw[300]+1; }
    ''', 'publish')
    scratch_kernel.compile(); publish_kernel.compile()
    modes = ('none', 'empty', 'scratch', 'publish')
    result = dict(scope=__doc__, psana=psana.__file__, cupy=cp.__version__,
                  cuda=cp.cuda.runtime.runtimeGetVersion(), device=str(cp.cuda.runtime.getDeviceProperties(0)['name']),
                  repetitions=a.repetitions, submissions=a.submissions, preflights=[], samples=[])
    a.output.parent.mkdir(parents=True, exist_ok=True)

    for batch_size in (1, 20):
        budget = _GpuBudget(64*1024**2)
        parser, initial, dtype = _setup(cp, budget=budget)
        producer = cp.cuda.Stream(non_blocking=True)
        batch, events, expected = _input(cp, parser, producer, [(0,)]*batch_size, dtype)
        producer.synchronize()
        specs = tuple(e.event for e in events)
        descriptors = batch._test_descriptors
        envelopes = [NS(dgrams=[NS(timestamp=lambda ts=e.timestamp: ts)]) for e in specs]
        gv = NS(iter_events=lambda: iter(specs))

        class ImmutableParsedInput:
            # The fixture owns immutable raw/parser arrays through all drains.
            # Closing one window releases its facade, not the shared fixture.
            def __getattr__(self, name): return getattr(batch, name)
            def retire(self): pass

        for depth in (1, 2):
            preparer = gd.DenseInputPreparer(initial.det_shape, initial.binding,
                                            n_slots=depth, budget=budget)
            preparer.configure_gather(parser.handle_indices)
            cp.cuda.get_current_stream().synchronize()

            def run_case(mode, submissions, *, check=False):
                pool = EventPool(n=depth)
                calls = 0
                def callback(evt, stream):
                    nonlocal calls
                    if check: calls += 1
                    if mode == 'empty': return
                    raw = evt.input('camera.raw')
                    if mode == 'scratch':
                        out = cp.empty_like(raw)
                        evt.keepalive(out)
                        scratch_kernel((4,), (256,), (raw, out), stream=stream)
                    else:
                        out = cp.empty((), cp.uint32)
                        evt.publish('value', out)
                        publish_kernel((1,), (1,), (raw, out), stream=stream)
                task = None if mode == 'none' else GpuTask(callback, ['camera.raw'])
                options = dict(task=task, detector_bindings={'camera': initial.binding})
                submit_ns = retire_ns = 0
                total_start = time.perf_counter_ns()
                for _ in range(submissions):
                    t = time.perf_counter_ns()
                    pool.begin_retire_next(); pool.finish_retire_next()
                    retire_ns += time.perf_counter_ns()-t
                    window = InputWindow(7, 0, ImmutableParsedInput(), descriptors)
                    try:
                        t = time.perf_counter_ns()
                        record = pool.submit(gv, None, envelopes, {'camera.raw': preparer},
                                             input_windows=(window,), batch_id=7, **options)
                        submit_ns += time.perf_counter_ns()-t
                    finally:
                        window.close()
                if check:
                    record.stream.synchronize()
                    np.testing.assert_array_equal(record.prepared_inputs['camera.raw'].data.get(), np.asarray(expected))
                    assert calls == (0 if mode == 'none' else batch_size*submissions)
                    if mode == 'publish':
                        assert len(record.publications_by_ts) == batch_size
                        for i,e in enumerate(specs):
                            value=record.publications_by_ts[e.timestamp]['value']
                            assert value.shape == () and value.nbytes == 4
                            assert int(cp.asnumpy(value.array)) == int(expected[i].reshape(-1)[300])+1
                    else:
                        assert not record.publications_by_ts
                    if mode == 'scratch':
                        arrays=[x for x in record.producer_owners if isinstance(x,cp.ndarray)]
                        assert len(arrays)==batch_size
                        for x,ref in zip(arrays,expected): np.testing.assert_array_equal(x.get(),ref+1)
                t=time.perf_counter_ns()
                for _ in pool.flush(): pass
                retire_ns += time.perf_counter_ns()-t
                total_ns=time.perf_counter_ns()-total_start
                assert not pool.active_count
                assert batch._locators == {}
                return dict(mode=mode, batch_size=batch_size, depth=depth,
                            events=batch_size*submissions, submit_ns=submit_ns,
                            retire_ns=retire_ns, loop_ns=total_ns,
                            submit_us_per_event=submit_ns/(batch_size*submissions)/1000,
                            loop_us_per_event=total_ns/(batch_size*submissions)/1000,
                            budget_committed_after_drain=budget.committed(),
                            cupy_pool_used_after_drain=cp.get_default_memory_pool().used_bytes())

            for mode in modes:
                # Warm allocations/kernels before separate counted correctness.
                run_case(mode, 5)
                counts = dict(walk=0, init=0, locate=0, gather=0)
                originals=[]
                for module,name,label in [(parser_module,'_walk_kernel','walk'),
                        (parser_module,'_init_locators_kernel','init'),
                        (parser_module,'_locate_fields_kernel','locate'),
                        (gd,'_batched_gather_kernel','gather')]:
                    original=getattr(module,name); originals.append((module,name,original))
                    def factory(*args, _original=original, _label=label):
                        kernel=_original(*args)
                        def launch(*args,**kwargs):
                            counts[_label]+=1
                            return kernel(*args,**kwargs)
                        return launch
                    setattr(module,name,factory)
                try: run_case(mode,1,check=True)
                finally:
                    for module,name,original in originals: setattr(module,name,original)
                assert counts == dict(walk=0,init=0,locate=0,gather=1), counts
                result['preflights'].append(dict(mode=mode,batch_size=batch_size,depth=depth,launches=counts))
            for rep in range(1,a.repetitions+1):
                for mode in (modes if rep%2 else tuple(reversed(modes))):
                    row=run_case(mode,a.submissions); row['repetition']=rep
                    result['samples'].append(row)
                    a.output.write_text(json.dumps(result,indent=2)+'\n')
            preparer.trim_slot_buffers()
        batch.retire()
    result['summary']=[]
    for batch_size in (1,20):
        for depth in (1,2):
            for mode in modes:
                rows=[r for r in result['samples'] if (r['batch_size'],r['depth'],r['mode'])==(batch_size,depth,mode)]
                result['summary'].append(dict(batch_size=batch_size,depth=depth,mode=mode,
                    median_submit_us_per_event=statistics.median(r['submit_us_per_event'] for r in rows),
                    median_loop_us_per_event=statistics.median(r['loop_us_per_event'] for r in rows)))
    result['complete']=True
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('CALLBACK_COST_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
