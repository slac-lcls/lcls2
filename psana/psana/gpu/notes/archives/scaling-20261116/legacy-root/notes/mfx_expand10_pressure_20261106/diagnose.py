"""Bounded scalar-only observations of input admission; no scheduling changes."""
import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
import hashlib
import importlib
import json
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

PREP = Path(__file__).resolve().parents[1]/'mfx_expand10_performance_20261106'
sys.path.insert(0,str(PREP))
import runner


def group_info(group):
    return dict(stream=int(group.stream_id), group=int(group.group_id), small=bool(group.small),
                bytes=int(group.size), events=len(group.dgrams))


def plan_state(pool, groups, result):
    """Observe AFTER the real plan_slots poll, without another poll or wait."""
    live = list(pool._groups.values())
    free = pool.reader._n_slots-len({s.slot for s in live})
    small = set(pool._small)
    conflicts = []
    for group in groups:
        if group.small:
            if group.stream_id in small:
                conflicts.append(int(group.stream_id))
            small.add(group.stream_id)
    return dict(accepted=result is not None, free_raw_slots=free, requested_groups=len(groups),
                small_stream_conflicts=conflicts, requests=[group_info(g) for g in groups],
                live=[dict(key=list(s.key), slot=int(s.slot), planned=len(s.planned),
                           window_references=None if s.window is None else s.window.references,
                           **group_info(s.group)) for s in live])


def manager_state(manager):
    budget = manager._gpu_budget
    categories = Counter()
    for allocation in budget.allocation_snapshot():
        categories[allocation['category']] += allocation['capacity']
    return dict(limit_bytes=budget.limit(), committed_bytes=budget.committed(),
                held_bytes=budget._held, available_bytes=budget.available(),
                margin_bytes=manager._admission_margin, categories=dict(categories),
                active_executions=manager.event_pool.active_count,
                occupied_execution_slots=[i for i,s in enumerate(manager.event_pool._slots) if s is not None],
                next_execution_slot=manager.event_pool.next_slot_id)


class Trace:
    def __init__(self, max_examples=4):
        self.max_examples = max_examples
        self.attempts = self.successes = self.hold_calls = self.hold_failures = 0
        self.failures = Counter()
        self.examples, self.success_examples = [], []
        self.request_kinds = Counter()
        self.max_committed = self.max_held = self.max_hold_request = 0
        self.last_plan = None

    @contextmanager
    def observe(self):
        from psana.gpu.gpu_events import GpuEventManager
        from psana.gpu.gpu_input_group import InputGroupPool
        from psana.gpu.gpu_budget import _GpuBudget, GpuMemoryPressureError
        issue, plan, hold = GpuEventManager._issue_gpu_read, InputGroupPool.plan_slots, _GpuBudget.hold
        def observed_plan(pool, groups):
            result = plan(pool, groups)
            self.last_plan = plan_state(pool, groups, result)
            for group in groups:
                self.request_kinds['small' if group.small else 'large'] += 1
            return result
        def observed_hold(budget, n, **kwargs):
            self.hold_calls += 1
            self.max_hold_request = max(self.max_hold_request, int(n))
            try:
                return hold(budget,n,**kwargs)
            except GpuMemoryPressureError:
                self.hold_failures += 1
                raise
        def observed_issue(manager, subbatch, slot_id):
            self.attempts += 1
            self.last_plan = None
            phase = manager._gpu_task.function.window.phase
            before = manager_state(manager)
            self.max_committed = max(self.max_committed,before['committed_bytes'])
            self.max_held = max(self.max_held,before['held_bytes'])
            identity = dict(phase=phase,batch_id=manager._input_batch_id,
                            start=subbatch._start,stop=subbatch._end,slot=int(slot_id))
            try:
                result = issue(manager,subbatch,slot_id)
            except GpuMemoryPressureError as exc:
                self.failures[(phase,str(exc))] += 1
                if sum(row['phase']==phase for row in self.examples) < self.max_examples:
                    self.examples.append(dict(**identity,before=before,after=manager_state(manager),
                                              error=str(exc),plan=self.last_plan,
                                              traceback=[dict(file=f.filename,line=f.lineno,function=f.name)
                                                         for f in traceback.extract_tb(exc.__traceback__)]))
                raise  # Let the unchanged framework perform its normal retry.
            self.successes += 1
            if len(self.success_examples) < self.max_examples:
                self.success_examples.append(dict(**identity,before=before,after=manager_state(manager),plan=self.last_plan))
            return result
        with ExitStack() as stack:
            stack.enter_context(patch.object(InputGroupPool,'plan_slots',observed_plan))
            stack.enter_context(patch.object(_GpuBudget,'hold',observed_hold))
            stack.enter_context(patch.object(GpuEventManager,'_issue_gpu_read',observed_issue))
            yield self

    def report(self):
        return dict(attempts=self.attempts,successes=self.successes,hold_calls=self.hold_calls,
                    hold_failures=self.hold_failures,max_hold_request_bytes=self.max_hold_request,
                    max_committed_before_issue_bytes=self.max_committed,max_held_before_issue_bytes=self.max_held,
                    failures=[dict(phase=k[0],error=k[1],count=v) for k,v in sorted(self.failures.items())],
                    request_kinds_including_retries=dict(self.request_kinds),
                    failure_examples=self.examples,success_examples=self.success_examples,
                    interpretation='Scalar host snapshots only; no extra poll, CUDA wait, device allocation or lease acquisition. Diagnostics add Python overhead: rates are not performance acceptance.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--streams',type=int,choices=(1,2),required=True)
    p.add_argument('--budget-gib',type=int,choices=(4,8),default=4)
    p.add_argument('--target-bytes',type=int,choices=(8<<20,512<<10),default=8<<20)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    run_args=argparse.Namespace(root=runner.PILOT, output=args.output,mode='throughput',bins=100,
                              streams=args.streams,batch_size=7,warmup=100,state_cap_gib=27,reserve_gib=8,
                              verify_files=False,export=None,framework_budget_gib=args.budget_gib,
                              bulk_target_bytes=args.target_bytes)
    trace=Trace()
    status='fail'
    try:
        with trace.observe():runner.run(run_args)
        status='pass'
    finally:
        modules=('psana.gpu.gpu_events','psana.gpu.gpu_input_group','psana.gpu.gpu_stream',
                 'psana.gpu.gpu_group_schedule','psana.gpu.gpu_stream_read_plan','psana.gpu.gpu_budget',
                 'psana.gpu.gpu_input_window')
        paths=[Path(importlib.import_module(n).__file__) for n in modules]
        paths += [Path(__file__),Path(__file__).with_name('test_diagnose.py'),Path(__file__).with_name('run.sbatch')]
        result=dict(status=status,streams=args.streams,framework_budget_gib=args.budget_gib,
                    bulk_target_bytes=args.target_bytes,**trace.report(),
                    source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths})
        if args.output.is_dir():
            (args.output/'admission-trace.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
