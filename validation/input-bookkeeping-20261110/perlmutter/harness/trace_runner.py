"""Reuse the accepted runner and existing opt-in timer; no new CUDA waits."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'sim10_stream_length_20261109'
sys.path.insert(0, str(BASE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bds', type=int, choices=(1,), default=1)
    parser.add_argument('--events', type=int, choices=(10000,), default=10000)
    parser.add_argument('--streams', type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    # This performs the accepted rank/UUID mask setup BEFORE MPI/CuPy imports.
    runner = runpy.run_path(str(HERE / 'runner.py'), run_name='accepted_runner')
    recorder = None
    if runner['rank_hint'] == 2:
        import cupy as cp
        from psana.gpu.gpu_events import GpuEventManager
        sys.path.insert(0, str(HERE.parents[1] / 'psana/psana/gpu/scripts'))
        from bulk_phase_timing import install, measured_function
        import bulk_accounting

        capture = os.environ.get('SIM10_NSYS_CAPTURE') == '1'
        recorder = install('Integrated-on', cp.cuda.nvtx if capture else None)
        from psana.gpu.gpu_input_group import InputGroupPool
        import hashlib
        import inspect
        measured_methods = {'poll': 'input.poll', 'plan_slots': 'input.plan_slots'}
        if os.environ['SIM10_VARIANT']=='new':
            measured_methods.update(issue_groups='input.issue_groups', _issue_reserved='input.issue')
        else:
            measured_methods['issue']='input.issue'
        for method, phase in measured_methods.items():
            fn = getattr(InputGroupPool, method)
            recorder.inventory.append(dict(function=fn.__qualname__,
                path=inspect.getfile(fn), ranges=[phase],
                source_sha256=hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()))
        poll = InputGroupPool.poll
        poll_counts = dict(calls=0, retained_states_at_poll=0, max_states=0, reclaimed=0)
        recorder.counters['input_poll'] = poll_counts

        def observed_poll(pool):
            # O(1) accounting only; do not add another scan or CUDA query.
            if recorder.active:
                size = len(pool._groups)
                poll_counts['calls'] += 1
                poll_counts['retained_states_at_poll'] += size
                poll_counts['max_states'] = max(poll_counts['max_states'], size)
            with recorder.phase('input.poll'):
                result = poll(pool)
            if recorder.active:
                poll_counts['reclaimed'] += len(result)
            return result

        InputGroupPool.poll = observed_poll
        InputGroupPool.plan_slots = measured_function(
            InputGroupPool.plan_slots, recorder, 'input.plan_slots')
        if os.environ['SIM10_VARIANT']=='new':
            InputGroupPool._issue_reserved = measured_function(
                InputGroupPool._issue_reserved, recorder, 'input.issue')
            InputGroupPool.issue_groups = measured_function(
                InputGroupPool.issue_groups, recorder, 'input.issue_groups')
        else:
            InputGroupPool.issue = measured_function(
                InputGroupPool.issue, recorder, 'input.issue')
        task = runner['StagingTask']
        task.__call__ = measured_function(task.__call__, recorder, 'task.callback')
        attach = GpuEventManager._attach_gpu

        def delivered(*a, **kw):
            result = attach(*a, **kw)
            recorder.steady = True
            return result

        GpuEventManager._attach_gpu = delivered
        original_accounting = bulk_accounting.accounting

        @contextmanager
        def accounting():
            # Entered after the accepted zero-resident-page gate and barrier.
            if capture:
                cp.cuda.profiler.start()
            recorder.active = True
            try:
                with original_accounting() as counters:
                    with recorder.phase('bd.loop'):
                        yield counters
            finally:
                recorder.active = False
                if capture:
                    cp.cuda.profiler.stop()

        bulk_accounting.accounting = accounting
    try:
        runner['main'](args)
        # Write timing evidence only after the accepted loop/cleanup/barriers.
        if recorder is not None:
            Path(args.output + '.phases.json').write_text(
                json.dumps(recorder.snapshot(), indent=2) + '\n')
    except BaseException:
        import traceback
        traceback.print_exc()
        runner['comm'].Abort(1)


if __name__ == '__main__':
    main()
