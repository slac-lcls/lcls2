"""Three-mode schedule, isolation, and sealed evidence; no native/GPU launch."""
import copy
from collections import Counter
import itertools
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import overhead as o
import native_contract as n
import test_native as native_fixtures
import test_overhead as fixtures


class ABCContractTests(unittest.TestCase):
    def setUp(self):
        self.fixture=native_fixtures.NativeTests();self.fixture.setUp()
    def tearDown(self):self.fixture.tearDown()

    def test_baseline_requires_abc_schema_and_overhead_phase(self):
        request=o.make_request(self.fixture.config,o.abc_schedule(self.fixture.config['cases'],True)[0])
        self.assertEqual(request['contract']['instrumentation'],'baseline')
        n.load_contract(request)
        for change in (dict(schema=o.SCHEMA),dict(schema=n.SCHEMA),dict(phase='aa')):
            with self.subTest(change=change),self.assertRaises(ValueError):
                n.load_contract(dict(request,**change))
        for item in o.abc_schedule(self.fixture.config['cases'],True):
            n.load_contract(o.make_request(self.fixture.config,item))

    def test_baseline_cannot_claim_population_or_details(self):
        rows=self.fixture.rows(1,True)
        fields=('event_count','timestamps','steps','cpu_raw_bytes','io','delivery','first_batch_seconds','callback_events','callback_sizes')
        for row in rows:
            for key in fields:row[key]=None
        case=dict(bds=1,workload='gpu-raw',iterator='events')
        self.assertEqual(n.check_baseline_rows(rows,[],case),dict(events=None,event_sha256=None,requested_bytes=None))
        for key in fields:
            bad=copy.deepcopy(rows);bad[2][key]=0
            with self.subTest(key=key),self.assertRaises(ValueError):n.check_baseline_rows(bad,[],case)
        bad=copy.deepcopy(rows);bad[2]['physical_cores']=bad[0]['physical_cores']
        with self.assertRaisesRegex(ValueError,'overlapping'):n.check_baseline_rows(bad,[],case)
        with self.assertRaises(ValueError):n.check_baseline_rows(rows[:-1],[],case)


class ABCScheduleTests(unittest.TestCase):
    def test_all_six_orders_twice_and_equal_positions(self):
        cases=n.read(Path(o.__file__).with_name('profile.json'))['cases']
        samples=o.abc_schedule(cases,False)
        self.assertEqual(len(samples),36)
        self.assertEqual(len({s['id'] for s in samples}),36)
        self.assertEqual(Counter(o.ABC_ORDERS),Counter({''.join(p):2 for p in itertools.permutations('ABC')}))
        for position in range(3):
            self.assertEqual(Counter(order[position] for order in o.ABC_ORDERS),Counter(A=4,B=4,C=4))
        for block in range(12):
            self.assertEqual({s['mode'] for s in samples[3*block:3*block+3]},set(o.ABC_MODES.values()))
        self.assertEqual(len(o.abc_schedule(cases,True)),6)

    def test_ratios_are_within_round_not_ratios_of_overall_means(self):
        timings={}
        for r in range(12):
            for mode,factor in (('baseline',1),('minimal',1.02),('full',1.03)):
                # Large common drift should cancel when comparing within rounds.
                timings[('gpu-raw-events-bd1',r,mode)]={k:100*(r+1)*factor for k in ('loop_seconds','setup_seconds','process_seconds')}
        result=o.abc_summary(timings,dict(pilot=False),[])
        for comparison,expected in (('B_over_A',2),('C_over_A',3),('C_over_B',100*(1.03/1.02-1))):
            row=result['comparisons'][comparison]['loop_seconds']
            self.assertAlmostEqual(row['mean_percent_change'],expected)
            self.assertEqual(row['pairs'],12)
        self.assertTrue(result['screen_passed']);self.assertFalse(result['overhead_qualified'])
        self.assertFalse(o.abc_summary(timings,dict(pilot=True),[])['screen_passed'])
        del timings[('gpu-raw-events-bd1',11,'baseline')]
        with self.assertRaises(KeyError):o.abc_summary(timings,dict(pilot=False),[])


class ABCSealedTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.SealedCaptureTests();self.fixture.setUp()
    def tearDown(self):self.fixture.tearDown()

    def prepare(self,pilot):
        f=self.fixture
        if not pilot:f.full_plan()
        f.plan.update(schema=o.ABC_SCHEMA,pilot=pilot,margin_percent=5.0,samples=o.abc_schedule(f.config['cases'],pilot))
        (f.root/'plan.json').write_text(json.dumps(f.plan))
        original=f.capture
        def capture(request,stage,folder):
            if request['contract']['instrumentation']!='baseline':return original(request,stage,folder)
            minimal=copy.deepcopy(request);minimal['contract']['instrumentation']='minimal'
            complete=original(minimal,stage,folder)
            for row in complete['rows']:
                row['instrumentation']='baseline';row['event_count']=None;row['callback_events']=None
            complete['work']=n.check_baseline_rows(complete['rows'],f.reference['events'],request['case'])
            return complete
        with patch.object(f,'capture',side_effect=capture):f.seal()

    def test_36_samples_seal_and_preserve_unobserved_baseline(self):
        self.prepare(False);f=self.fixture
        result=o.collect(f.root)
        self.assertEqual(len(result['samples']),36)
        self.assertTrue(result['screen_passed']);self.assertFalse(result['overhead_qualified'])
        self.assertIsNone(result['mode_timings'][0]['modes']['A']['events'])
        self.assertEqual(result['mode_timings'][0]['modes']['B']['events'],10000)
        self.assertTrue(result['mode_timings'][0]['modes']['C']['exact_identity_and_bytes_checked'])
        with self.assertRaisesRegex(ValueError,'own three-mode'):o.screen(f.root)
        (f.root/'samples'/f.plan['samples'][0]['id']/'measured/complete.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'capture missing/changed'):o.collect(f.root)

    def test_pilot_is_matching_design_only_and_does_not_pass_screen(self):
        self.prepare(True);f=self.fixture
        result=o.collect(f.root)
        self.assertEqual(len(result['samples']),6)
        self.assertFalse(result['screen_passed'])
        sha=n.file_hash(f.root/'complete.json')
        o.check_pilot(f.root,sha,{}, {},o.ABC_SCHEMA)
        with self.assertRaisesRegex(ValueError,'matching-design'):o.check_pilot(f.root,sha,{}, {})
        with self.assertRaisesRegex(ValueError,'source/inputs'):o.check_pilot(f.root,sha,{'x':'y'}, {},o.ABC_SCHEMA)

    def test_reordered_or_missing_sample_fails_even_before_sealing(self):
        self.prepare(True);f=self.fixture
        f.plan['samples'][0],f.plan['samples'][1]=f.plan['samples'][1],f.plan['samples'][0]
        (f.root/'plan.json').write_text(json.dumps(f.plan))
        with self.assertRaisesRegex(ValueError,'schedule changed'):o._collect_samples(f.root)


if __name__=='__main__':unittest.main()
