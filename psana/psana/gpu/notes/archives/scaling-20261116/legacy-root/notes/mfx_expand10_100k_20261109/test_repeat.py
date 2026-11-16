import unittest
from repeat import timeline
from validate_repeat import reference_plan
from xtc_fixture import from_ns,to_ns


def rows():
    return [{'timestamp':from_ns(t),'service':s} for t,s in [(900000000,2),(910000000,4),(920000000,6),(930000000,8),(940000000,10),(950000000,12),(960000000,12),(970000000,10),(980000000,9),(990000000,7),(999999999,5)]]


class RepeatTests(unittest.TestCase):
    def test_independent_schedule(self):
        for count in (1,2,3,4,5,23,100000):
            actual,period=timeline(rows(),count,20000000)
            expected,ref_period=reference_plan(rows(),count,20000000)
            self.assertEqual(period,ref_period)
            self.assertEqual([(x['source_index'],x['cycle'],x['timestamp'],x['service']) for x in actual],expected)

    def test_partial_and_rollover(self):
        plan,period=timeline(rows(),5,20000000)
        self.assertEqual(period,50000000)
        self.assertEqual([p['source_index'] for p in plan if p['service']==12],[5,6,5,6,5])
        self.assertEqual([p['output_event_index'] for p in plan if p['service']==12],list(range(5)))
        times=[to_ns(p['timestamp']) for p in plan]
        self.assertTrue(all(a<b for a,b in zip(times,times[1:])))
        self.assertGreater(times[-1],1000000000)
        self.assertEqual([p['service'] for p in plan if p['service'] not in (10,12)],[2,4,6,8,9,7,5])

    def test_complete_cycle_keeps_trailing_update(self):
        plan,_=timeline(rows(),4)
        self.assertEqual([p['service'] for p in plan][-4:],[10,9,7,5])
        self.assertEqual(sum(p['service']==10 for p in plan),4)

    def test_bad_source_and_counts(self):
        for count in (0,-1,100001,True):
            with self.assertRaises(ValueError):timeline(rows(),count)
        for gap in (0,-1,True):
            with self.assertRaises(ValueError):timeline(rows(),1,gap)
        broken=rows();broken[5]['timestamp']=broken[4]['timestamp']
        with self.assertRaises(ValueError):timeline(broken,1)
        broken=rows();broken[4]['service']=7
        with self.assertRaises(ValueError):timeline(broken,1)


if __name__=='__main__':unittest.main()
