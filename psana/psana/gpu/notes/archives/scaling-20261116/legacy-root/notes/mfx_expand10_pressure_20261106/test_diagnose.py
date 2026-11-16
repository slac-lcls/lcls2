"""Verify snapshots against the real input-credit planner without a GPU."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from diagnose import Trace, plan_state
from psana.gpu.gpu_input_group import InputGroupPool


def group(index, stream=5, small=True):
    return NS(group_id=index,stream_id=stream,small=small,size=600000,dgrams=(object(),))


def pool(slots=4):
    reader=NS(bulk_read=True,_n_slots=slots,_pending=[],_input_holds={},_slot_bufs=[None]*slots)
    return InputGroupPool(reader)


class Tests(unittest.TestCase):
    def test_live_small_credit_blocks_despite_free_slots(self):
        p=pool(); old=group(0); new=group(1)
        state=NS(key=(1,0),group=old,slot=0,planned={},window=None)
        p._groups[state.key]=state;p._small[5]=state.key
        result=p.plan_slots((new,));self.assertIsNone(result)
        snap=plan_state(p,(new,),result)
        self.assertEqual(snap['free_raw_slots'],3)
        self.assertEqual(snap['small_stream_conflicts'],[5])
        # Matching one-slot retirement releases its credit before the next read.
        p._groups.clear();p._small.clear()
        self.assertIsNotNone(p.plan_slots((new,)))
        self.assertEqual(snap['live'][0]['stream'],5)  # Snapshot owns scalars only.

    def test_large_groups_do_not_share_small_credit_and_no_slots_distinct(self):
        p=pool();g=group(0,small=False)
        p._groups[(1,0)]=NS(key=(1,0),group=g,slot=0,planned={},window=None)
        self.assertIsNotNone(p.plan_slots((group(1,small=False),)))
        p.reader._n_slots=1;p.reader._slot_bufs=[None]
        result=p.plan_slots((group(1,small=False),));self.assertIsNone(result)
        snap=plan_state(p,(group(1,small=False),),result)
        self.assertEqual(snap['free_raw_slots'],0);self.assertEqual(snap['small_stream_conflicts'],[])

    def test_observer_forwards_and_restores_on_failure(self):
        from psana.gpu.gpu_events import GpuEventManager
        from psana.gpu.gpu_budget import _GpuBudget,GpuMemoryPressureError
        originals=(GpuEventManager._issue_gpu_read,InputGroupPool.plan_slots,_GpuBudget.hold)
        trace=Trace()
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with trace.observe():
                self.assertEqual(pool().plan_slots((group(1),)),(0,))
                b=_GpuBudget(100)
                h=b.hold(20);h.close()
                with self.assertRaises(GpuMemoryPressureError):b.hold(101)
                raise RuntimeError('injected')
        self.assertEqual(trace.hold_calls,2);self.assertEqual(trace.hold_failures,1)
        self.assertEqual(originals,(GpuEventManager._issue_gpu_read,InputGroupPool.plan_slots,_GpuBudget.hold))

    def test_issue_error_is_preserved_with_budget_and_credit_snapshot(self):
        from psana.gpu.gpu_events import GpuEventManager
        from psana.gpu.gpu_budget import _GpuBudget,GpuMemoryPressureError
        p=pool();g=group(0)
        p._groups[(1,0)]=NS(key=(1,0),group=g,slot=0,planned={},window=None)
        p._small[5]=(1,0)
        error=GpuMemoryPressureError('input groups await stream credit or raw slots')
        def rejected(manager,batch,slot):
            self.assertIsNone(p.plan_slots((group(1),)))
            raise error
        manager=NS(_gpu_budget=_GpuBudget(4<<30),_admission_margin=(4<<30)//10,
                   event_pool=NS(active_count=1,_slots=[object(),None],next_slot_id=1),
                   _gpu_task=NS(function=NS(window=NS(phase='steady'))),_input_batch_id=2)
        trace=Trace(max_examples=1)
        with patch.object(GpuEventManager,'_issue_gpu_read',rejected),trace.observe():
            for _ in range(2):
                with self.assertRaises(GpuMemoryPressureError) as caught:
                    GpuEventManager._issue_gpu_read(manager,NS(_start=0,_end=7),1)
                self.assertIs(caught.exception,error)
        result=trace.report()
        self.assertEqual(result['failures'][0]['count'],2)
        self.assertEqual(len(result['failure_examples']),1)
        self.assertEqual(result['failure_examples'][0]['plan']['small_stream_conflicts'],[5])
        self.assertEqual(result['failure_examples'][0]['after']['available_bytes'],4<<30)


if __name__=='__main__':unittest.main()
