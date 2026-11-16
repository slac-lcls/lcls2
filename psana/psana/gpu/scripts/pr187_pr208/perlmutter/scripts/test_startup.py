"""CPU tests of measurement boundaries, exception semantics and fixed statistics."""
import json
from pathlib import Path
import tempfile
import unittest
from startup_timing import Recorder, FirstBatch
from schedule import schedule, validate
from analysis import summarize, METRICS


class Tests(unittest.TestCase):
    def test_boundary_waits_for_gpu_and_last_delivery_only_once(self):
        calls=[]
        class Event:
            def __init__(self, disable_timing):
                self.complete=False
                self.assert_disabled=disable_timing
            def record(self, stream): calls.append(('record',stream))
            def synchronize(self):
                calls.append('sync');self.complete=True
            @property
            def done(self):return self.complete
        boundary=FirstBatch()
        boundary.submitted(range(20),'work-stream',Event)
        for n in range(1,20):
            self.assertFalse(boundary.delivered(list(range(n)),10,lambda:12))
        self.assertTrue(boundary.delivered(list(range(20)),10,lambda:12))
        self.assertEqual((boundary.seconds,boundary.events),(2,20))
        boundary.submitted(range(20,40),'other-stream',Event)
        self.assertFalse(boundary.delivered(list(range(21)),10,lambda:13))
        self.assertEqual(calls,[('record','work-stream'),'sync'])

    def test_boundary_rejects_mismatched_delivery(self):
        boundary=FirstBatch();boundary.timestamps=tuple(range(20))
        with self.assertRaises(AssertionError):boundary.delivered([1,19],0)

    def test_wrappers_preserve_return_and_exception(self):
        sentinel=object()
        class Example:
            def call(self, failure=False):
                if failure:raise ValueError('original')
                return sentinel
        clock=iter([1.,3.,5.,8.])
        rec=Recorder(clock=lambda:next(clock))
        rec.wrap(Example,'call',byte_count=lambda a,k:7)
        self.assertIs(Example().call(),sentinel)
        rec.phase='loop'
        with self.assertRaisesRegex(ValueError,'original'):Example().call(True)
        self.assertEqual(rec.records['Example.call'],dict(count=2,total_s=5.,max_s=3.,bytes=14,failures=1))
        self.assertEqual(rec.records['loop/Example.call']['count'],1)

    def test_schedule_balance_and_rejection(self):
        plan=validate(schedule())
        self.assertEqual(plan,schedule())
        bad=json.loads(json.dumps(plan));bad[2]['order']=bad[0]['order']
        with self.assertRaises(AssertionError):validate(bad)

    def fixture(self, output):
        plan=schedule()
        (output/'provenance.json').write_text(json.dumps(dict(schedule=plan)))
        rows=[]
        for block in plan:
            for case in block['order']:
                value=(10+block['repetition']**2)*{'A':1,'B-off':1.1,'B-on':1.21}[case]
                bd=dict(rank=2,role='bd',events=250,setup_s=value,first_batch_s=value,
                    remaining_s=value,remaining_events=230,loop_s=value,last_event_s=value,
                    tail_drain_s=value,timers={},snapshots=[],callback_sizes={'20':10})
                row=dict(events=block['events'],repetition=block['repetition'],case=case,
                         diagnostic=False,trace=False,tag=str(len(rows)),rows=[bd])
                row.update({key:value for key in METRICS})
                rows.append(row)
        (output/'results.json').write_text(json.dumps(rows))
        return rows

    def test_analysis_pairs_cancel_common_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);self.fixture(output)
            s=summarize(output)
            self.assertEqual(s['clean_samples'],36)
            for comparison in s['comparisons']:
                for value in comparison['metrics'].values():
                    self.assertAlmostEqual(value['percent']['mean'],10)
                    self.assertFalse(value['percent']['within_5pct'])

    def test_missing_duplicate_or_reordered_samples_rejected(self):
        for mode in ('missing','duplicate','reordered'):
            with tempfile.TemporaryDirectory() as directory:
                output=Path(directory);rows=self.fixture(output)
                if mode=='missing':rows.pop()
                elif mode=='duplicate':rows[-1]=rows[-2]
                else:rows[0],rows[1]=rows[1],rows[0]
                (output/'results.json').write_text(json.dumps(rows))
                with self.assertRaises(AssertionError):summarize(output)


if __name__=='__main__':unittest.main()
