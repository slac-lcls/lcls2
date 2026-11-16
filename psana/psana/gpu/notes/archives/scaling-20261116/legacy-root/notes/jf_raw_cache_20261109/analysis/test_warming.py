import json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common

def state(present):
    return dict(files=[dict(path='f',bytes=40960,pages=10,resident_pages=present)],pages=10,resident_pages=present)
class Warming(unittest.TestCase):
    def test_failed_preparation_preserves_all_checks(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'cache.json'
            with patch.dict(os.environ,JF_CACHE_REPORT=str(out)),patch('memory_state.snapshot',return_value={}),patch.object(common,'paths',return_value=[]),patch.object(common,'residency',side_effect=[state(0)]+[state(9)]*9),patch.object(common.subprocess,'run'):
                with self.assertRaisesRegex(ValueError,'Cache gate failed'):common.condition('unused','warm')
            r=json.loads(out.read_text());self.assertEqual(r['status'],'fail');self.assertEqual(len(r['history']),10);self.assertEqual(r['cache']['resident_pages'],9)
    def test_repair_can_pass_without_lowering_gate(self):
        with patch('memory_state.snapshot',return_value={}),patch.object(common,'paths',return_value=[]),patch.object(common,'residency',side_effect=[state(0),state(9),state(10)]),patch.object(common.subprocess,'run') as run,patch.dict(os.environ,{},clear=True):
            r=common.condition('unused','warm')
            self.assertEqual(r['status'],'pass');self.assertEqual(run.call_count,2);self.assertIn('--interleave=all',run.call_args_list[0].args[0]);self.assertIn('--missing-prefixes',run.call_args_list[1].args[0])
