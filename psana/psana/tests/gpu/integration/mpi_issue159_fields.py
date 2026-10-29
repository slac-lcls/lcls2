"""Six-rank field/IPC lifetime and reporting acceptance for issue159.

Uses real psana MPI event delivery, placement, constants, budgets and input
leases. Synthetic packed fields guarantee misalignment independently of the
dataset; their source/output fixture buffers are outside the manager budget.
Real Jungfrau calibration pixel equality is checked by a separate driver.
"""
import argparse
from contextlib import closing
import gc
import json
import os
from pathlib import Path
import sys
import weakref

import numpy as np


def snapshot(manager, label):
    budget = manager._gpu_budget
    before = (budget.limit(), budget.committed(), budget._held)
    value = manager._snapshot_memory(label)
    assert before == (budget.limit(), budget.committed(), budget._held)
    return dict(label=label, limit=budget.limit(), committed=value.committed,
                held=value.held, constants=value.constants, borrowed=value.borrowed,
                retained=value.retained, failed=value.failed,
                field_allocations=[a for a in value.allocations
                                   if a['category'] == 'field-copy'],
                constant_allocations=[a for a in value.allocations
                                      if a['category'] == 'task-constants'])


def arm_fields(event, manager):
    import cupy as cp
    from psana.gpu.gpu_input import GpuFieldResult
    from psana.gpu.gpu_allocation import backing_capacity
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from field_view_fixture import field_case, retire

    # Fixture lease is unused: attach its fields to the REAL delivered event's
    # root lease, so the manager's real retirement must join these consumers.
    case = field_case(cp, type_id=3, shapes=((257,), (), (5,)), absent=(17,))
    case.lease.result_ready.synchronize()
    retire(case)
    lease = event.gpu._input_lease
    assert lease._budget is manager._gpu_budget and lease._owners
    first = GpuFieldResult(case.result.binding, case.sources, lease)
    second = GpuFieldResult(case.result.binding, case.sources, lease)
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    done = [cp.cuda.Event(disable_timing=True) for _ in streams]
    outputs = [{s: cp.empty(case.expected[s].shape, cp.uint64) for s in (9, 4)}
               for _ in streams]
    kernel = cp.RawKernel(r'''
    extern "C" __global__ void consume(const unsigned long long* src,
        const unsigned long long* scalar, const unsigned long long* vector,
        unsigned long long* dst, int n, unsigned long long ticks) {
        unsigned long long start=clock64();
        while(clock64()-start<ticks) {}
        for(int i=threadIdx.x; i<n; i+=blockDim.x)
            dst[i]=(src[i]^scalar[0])+vector[1];
    }''', 'consume')
    kernel.compile()
    scalar = manager._task_constants.get('jungfrau', 'alignment_scalar')
    vector = manager._task_constants.get('jungfrau', 'alignment_vector')
    empty = manager._task_constants.get('jungfrau', 'alignment_empty')
    assert scalar.shape == () and empty.shape == (0, 3) and empty.nbytes == 0
    np.testing.assert_array_equal(vector.get(), np.arange(16, dtype=np.uint64))
    assert scalar.get() == 7
    # Warm allocation and kernel loading before asserting asynchronous work.
    warm = [cp.empty((257,), cp.uint64), cp.empty((), cp.uint64)]
    warm[0].fill(0)
    kernel((1,), (32,), (warm[0], scalar, vector, outputs[0][9],
                        np.int32(257), np.uint64(0)), stream=streams[0])
    cp.cuda.Device().synchronize()
    del warm
    baseline = snapshot(manager, 'before-field')
    assert not baseline['field_allocations']
    contexts, refs, pointers, ready_ids = [], [], {}, ()
    copies = []
    memcpy = cp.cuda.runtime.memcpyAsync

    def counted(*args):
        copies.append((int(args[0]), int(args[2])))
        return memcpy(*args)

    cp.cuda.runtime.memcpyAsync = counted
    try:
        # Delay the first copy on its stream. The second stream gets no
        # explicit wait here: cached-field readiness must supply that edge.
        kernel((1,), (32,), (outputs[0][9], scalar, vector, outputs[0][9],
                            np.int32(0), np.uint64(2000000000)), stream=streams[0])
        for index, (field, stream) in enumerate(zip((first, second), streams)):
            context = field.on_gpu_view(stream)
            contexts.append(context)
            values = context.__enter__()
            assert values.segment_ids == (9, 4) and values[4].shape == ()
            if index == 0:
                pointers = {s: v.data.ptr for s, v in values.items()}
                assert all(ptr % 8 == 0 for ptr in pointers.values())
                refs = [weakref.ref(v) for v in values.values()]
                charge = sum(backing_capacity(v) for v in values.values())
                ready_ids = tuple(id(e) for e in first._views_ready)
                assert len(copies) == 2 and not all(e.done for e in first._views_ready)
            else:
                assert pointers == {s: v.data.ptr for s, v in values.items()}
                assert ready_ids == tuple(id(e) for e in second._views_ready)
                assert len(copies) == 2
            for segment, value in values.items():
                kernel((1,), (32,), (value, scalar, vector, outputs[index][segment],
                                    np.int32(value.size), np.uint64(0)), stream=stream)
            stream.record(done[index])
            del value, values
            current = snapshot(manager, 'first-copy' if index == 0 else 'cache-hit')
            assert len(current['field_allocations']) == 2
            assert sum(a['capacity'] for a in current['field_allocations']) == charge
            assert current['committed'] - baseline['committed'] == charge
            assert current['retained'] - baseline['retained'] == charge
            assert (current['constants'], current['borrowed'], current['held']) == (
                baseline['constants'], baseline['borrowed'], baseline['held'])
        pending = [not e.done for e in done]
        assert all(pending), 'fixture did not leave both consumers pending'
    finally:
        cp.cuda.runtime.memcpyAsync = memcpy
        for context in reversed(contexts):
            context.__exit__(None, None, None)
    # No output read/synchronize here: advance/close must do the joining.
    after_context = snapshot(manager, 'contexts-exited')
    assert sum(a['capacity'] for a in after_context['field_allocations']) == charge
    return dict(case=case, fields=(first, second), lease=lease, outputs=outputs,
                streams=streams, done=done, refs=refs, baseline=baseline,
                charged=current, after_context=after_context, bytes=charge,
                copies=len(copies), pending=pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exit', choices=('normal', 'early'), required=True)
    parser.add_argument('--bulk', type=int, choices=(0, 1), required=True)
    parser.add_argument('--budget-gb', type=float, default=0)
    args = parser.parse_args()
    from mpi4py import MPI
    from psana import DataSource
    from psana.gpu import GpuTask
    from psana.gpu.gpu_shared_constants import SharedRequestedConstants
    from psana.gpu.gpu_task import RequestedConstants
    from psana.gpu.gpu_mpi import gpu_error_handler
    from psana.psexp.mpi_ds import RunParallel
    from psana.psexp.run import Run
    comm = MPI.COMM_WORLD
    assert comm.size == 6
    shared = os.environ['PSANA_GPU_SHARE_CONSTANTS'] == '1'
    host = {'alignment_scalar': np.array(7, np.uint64),
            'alignment_empty': np.empty((0, 3), np.uint64),
            'alignment_vector': np.arange(16, dtype=np.uint64)}
    constant_bytes = sum(v.nbytes for v in host.values())
    state = dict(manager=None, field=None, checks=[], close_checked=False,
                 host_transport_shapes={})
    original_setup, original_make = Run._setup_run_calibconst, RunParallel._make_gpu_event_manager
    original_closes = {cls: cls.close for cls in (RequestedConstants, SharedRequestedConstants)}
    original_error_exit = gpu_error_handler.__exit__

    def diagnostic_error_exit(handler, exc_type, exc, tb):
        if exc is not None and not isinstance(exc, GeneratorExit):
            import traceback
            traceback.print_exception(exc_type, exc, tb)
            sys.stderr.flush()
        return original_error_exit(handler, exc_type, exc, tb)

    gpu_error_handler.__exit__ = diagnostic_error_exit

    def setup(run):
        run._calib_const = {'jungfrau': {k: (v, {}) for k, v in host.items()}}
        run.dsparms.calibconst = run._calib_const

    def capture(run):
        # Host calibration XTC transport currently promotes NumPy scalars to
        # (1,) in CalibXtcConverter._build_datadef. Record that separate limit,
        # then inject the intended fixture at the GPU-manager boundary. The
        # real-data pixel driver does not patch this calibration path.
        for key in host:
            value = run.dsparms.calibconst['jungfrau'][key]
            if isinstance(value, tuple) and len(value) == 2:
                value = value[0]
            state['host_transport_shapes'][key] = list(value.shape)
        setup(run)
        manager = original_make(run)
        state['manager'] = manager
        return manager

    def wrap_close(original):
        def checked(constants):
            manager, probe = state['manager'], state['field']
            inspect = (manager is not None and constants is manager._task_constants
                       and probe is not None and not state['close_checked'])
            if inspect:
                assert all(e.done for e in probe['done']), 'IPC teardown preceded consumers'
                assert probe['lease']._closed
                gc.collect()
                assert all(ref() is None for ref in probe['refs'])
                before = snapshot(manager, 'before-constant-close')
                assert not before['field_allocations']
                state['checks'].append(before)
            original(constants)
            if inspect:
                after = snapshot(manager, 'after-constant-close')
                assert after['constants'] == after['borrowed'] == 0
                assert not after['field_allocations'] and not after['constant_allocations']
                assert after['failed'] == 0
                state['checks'].append(after)
                state['close_checked'] = True
        return checked

    Run._setup_run_calibconst, RunParallel._make_gpu_event_manager = setup, capture
    for cls, original in original_closes.items():
        cls.close = wrap_close(original)
    calls = []

    def callback(batch, stream):
        calls.append(batch.size)
        scalar_shape = batch.calibconst('jungfrau', 'alignment_scalar').shape
        empty_shape = batch.calibconst('jungfrau', 'alignment_empty').shape
        assert scalar_shape == (), ('callback scalar shape', scalar_shape)
        assert empty_shape == (0, 3), ('callback empty shape', empty_shape)
        batch.publish('timestamp', batch.timestamps_gpu)

    opts = dict(exp='mfx100848724', run=51, dir='/sdf/data/lcls/ds/prj/public01/xtc',
                detectors=['jungfrau'], gpu_det='jungfrau', max_events=40,
                batch_size=5, n_gpu_streams=2, gpu_bulk_read=bool(args.bulk),
                gpu_fn=GpuTask(callback, ['jungfrau.raw'],
                               [('jungfrau', key) for key in host]))
    if args.budget_gb:
        opts['gpu_memory_budget_gb'] = args.budget_gb
    info = dict(rank=comm.rank, mask=os.environ.get('CUDA_VISIBLE_DEVICES'),
                sharing_requested=shared, exit=args.exit, bulk=args.bulk,
                budget_gb=args.budget_gb, stamps=[])
    for run in DataSource(**opts).runs():
        info['role'] = run.comms.node_type()
        with closing(run.events()) as events:
            for event in events:
                info['stamps'].append(int(event.timestamp))
                if state['field'] is None:
                    import cupy as cp
                    manager = state['manager']
                    placement = manager._placement
                    assert placement.n_device_peers == 4
                    assert cp.cuda.Device().pci_bus_id.upper().endswith(
                        placement.device_pci.upper().split(':', 1)[1])
                    info.update(owner=placement.is_owner, pci=placement.device_pci,
                                peers=placement.n_device_peers, sharing=placement.sharing,
                                usable=placement.usable_bytes)
                    assert (placement.sharing == 'on') == shared
                    try:
                        state['field'] = arm_fields(event, manager)
                    except BaseException:
                        # Preserve the initiating assertion before an early
                        # close can produce a secondary teardown assertion.
                        import traceback
                        traceback.print_exc()
                        sys.stderr.flush()
                        raise
                    baseline = state['field']['baseline']
                    if shared:
                        assert baseline['constants'] == (constant_bytes if placement.is_owner else 0)
                        assert baseline['borrowed'] == (0 if placement.is_owner else constant_bytes)
                    else:
                        assert baseline['borrowed'] == 0
                        assert baseline['constants'] >= constant_bytes
                        assert baseline['constants'] == sum(a['capacity'] for a in baseline['constant_allocations'])
                    if args.budget_gb:
                        assert baseline['limit'] == int(args.budget_gb * 1024**3)
                    else:
                        overhead = constant_bytes if shared else 0
                        want = (placement.usable_bytes - overhead) // 4
                        if shared and placement.is_owner:
                            want += overhead
                        assert baseline['limit'] == want
                if args.exit == 'early':
                    break
        break
    if info['role'] == 'bd':
        probe, manager = state['field'], state['manager']
        assert probe is not None, 'BD never exercised a field'
        assert state['close_checked'] and manager._closed
        assert manager.event_pool.active_count == 0
        for output in probe['outputs']:
            for segment, array in output.items():
                expected = (probe['case'].expected[segment] ^ np.uint64(7)) + np.uint64(1)
                np.testing.assert_array_equal(array.get(), expected)
        info.update(baseline=probe['baseline'], charged=probe['charged'],
                    contexts_exited=probe['after_context'], close_checks=state['checks'],
                    copies=probe['copies'], pending=probe['pending'], calls=calls,
                    host_transport_shapes=state['host_transport_shapes'])
    else:
        assert 'cupy' not in sys.modules
    Run._setup_run_calibconst, RunParallel._make_gpu_event_manager = original_setup, original_make
    gpu_error_handler.__exit__ = original_error_exit
    for cls, original in original_closes.items():
        cls.close = original
    state.clear()
    if info['role'] == 'bd':
        del probe, manager
    gc.collect()
    rows = comm.gather(info, root=0)
    if comm.rank == 0:
        roles = {role: sum(r['role'] == role for r in rows) for role in ('smd0', 'eb', 'bd')}
        assert roles == dict(smd0=1, eb=1, bd=4), roles
        workers = [r for r in rows if r['role'] == 'bd']
        assert all(r['stamps'] for r in workers)
        stamps = [s for r in workers for s in r['stamps']]
        assert len(stamps) == len(set(stamps)) == (40 if args.exit == 'normal' else 4)
        assert len({r['pci'] for r in workers}) == 1
        assert sum(r['owner'] for r in workers) == 1
        print('MPI_ALIGNMENT_REPORT_OK ' + json.dumps(dict(roles=roles, ranks=rows)), flush=True)
    comm.Barrier()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        from mpi4py import MPI
        MPI.COMM_WORLD.Abort(1)
        raise
