import copy
import unittest
import tempfile
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_smoke as smoke
from native_contract import write, read

from run_smoke import derive_prefix


class SmokePrefixTests(unittest.TestCase):
    def setUp(self):
        self.reference = dict(schema='run51-raw-reference-v1', events=[[1,0,120],[2,0,120],[3,0,120]])
        self.plan = dict(schema='run51-input-plan-v1', reference=self.reference, streams=[])
        self.manifest = dict(schema='run51-private-input-v2', private_copy=True, files=[])
        for stream in (3,5,7,8,9):
            self.plan['streams'].append(dict(stream=stream, smd=dict(sha256='smd'),
                 events=[[1,0,0,24,12],[2,0,48,24,11],[3,0,72,24,12]]))
            self.manifest['files'].append(dict(stream=stream, smd=dict(sha256='smd'),
                 bigdata=dict(size=96,ranges=[[0,24],[48,48]],range_bytes=72,sha256='parent')))

    def test_prefix_excludes_transition_gap_and_drops_parent_hash(self):
        original = copy.deepcopy(self.manifest)
        manifest, reference = derive_prefix(self.manifest,self.plan,self.reference,2)
        self.assertEqual(reference['events'],self.reference['events'][:2])
        self.assertEqual(manifest['files'][0]['bigdata']['ranges'],[[0,24],[48,24]])
        self.assertEqual(sum(f['bigdata']['range_bytes'] for f in manifest['files']),240)
        self.assertNotIn('sha256',manifest['files'][0]['bigdata'])
        self.assertEqual(self.manifest,original)

    def test_rejects_inconsistent_parent_ranges_reference_and_streams(self):
        for kind in ('range','reference','identity','smd','stream'):
            manifest, plan, reference = copy.deepcopy((self.manifest,self.plan,self.reference))
            if kind == 'range': manifest['files'][0]['bigdata']['ranges']=[[0,96]]
            if kind == 'reference': reference['events'][0][2]=121
            if kind == 'identity': plan['streams'][1]['events'][0][0]=9
            if kind == 'smd': manifest['files'][0]['smd']['sha256']='changed'
            if kind == 'stream': manifest['files'][0]['stream']=4
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                derive_prefix(manifest,plan,reference,2)

    def test_rejects_invalid_prefix_length(self):
        for count in (0,4,True,1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                derive_prefix(self.manifest,self.plan,self.reference,count)


class SmokeDriverTests(unittest.TestCase):
    def test_final_step_audit_rejects_numeric_empty_foreign_or_duplicate(self):
        good='StepId=123.batch Name=batch Network=(null)\nStepId=123.extern Name=extern Network=(null)\n'
        with patch.object(smoke,'command_output',return_value=good):
            self.assertEqual(smoke.final_step_audit('123')['status'],'pass')
        for bad in ('',good+'StepId=123.1 Name=worker Network=(null)\n',good+good,'StepId=999.batch Name=batch Network=(null)\n'):
            with patch.object(smoke,'command_output',return_value=bad), self.assertRaises(ValueError):
                smoke.final_step_audit('123')

    def test_spec_and_prefix_association_reject_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            write(folder/'spec.json',{'value':1}); pin=smoke.artifact(folder/'spec.json')
            write(folder/'event-reference.json',{}); write(folder/'dataset-manifest.json',{})
            write(folder/'complete.json',dict(status='pass',parent={'value':2},events=1000,
                  manifest_sha256=smoke.file_hash(folder/'dataset-manifest.json'), reference_sha256=smoke.file_hash(folder/'event-reference.json')))
            with self.assertRaisesRegex(ValueError,'association'):
                smoke.verify_prefix(folder,{'value':1})
            (folder/'spec.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                smoke.pinned(pin)

    def test_orchestration_stops_on_bad_blockers_missing_nic_or_case_failure(self):
        for failure in ('blocker','nic','process'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                root=Path(temp); runtime=root/'runtime.json'
                write(runtime,dict(python=sys.executable,commit='a'*40,install='/fake'))
                spec=dict(runtime=smoke.artifact(runtime)); inputs=root/'inputs.json';write(inputs,spec)
                args=SimpleNamespace(inputs=str(inputs),inputs_sha256=smoke.file_hash(inputs),output=str(root/'output'))
                calls=[]
                def execute(argv,log,timeout):
                    calls.append(argv)
                    if '_prefix' in argv:
                        folder=Path(argv[-1]);folder.mkdir()
                        write(folder/'event-reference.json',dict(events=[[i+1,0,1] for i in range(1000)]))
                        write(folder/'dataset-manifest.json',dict(root='/fake',files=[]))
                        write(folder/'complete.json',dict(status='pass',parent=spec,events=1000,
                              manifest_sha256=smoke.file_hash(folder/'dataset-manifest.json'),reference_sha256=smoke.file_hash(folder/'event-reference.json')))
                        return 0
                    if failure=='process':return 1
                    write(Path(argv[-1]),dict(nic=dict(rx_bytes=None,tx_bytes=0)))
                    return 0
                def placement(options):write(Path(options.output),{})
                with patch.dict(smoke.os.environ,{'SLURM_JOB_ID':'123'}), patch.object(smoke.prepare_site,'capture',side_effect=placement), patch.object(smoke.campaign,'run_command',side_effect=execute), patch.object(smoke.campaign,'validate',return_value=[] if failure=='blocker' else ['native_adapter']), patch.object(smoke.campaign,'validate_result'):
                    with self.assertRaises(ValueError):smoke.run(args)
                self.assertEqual(len(calls),1 if failure=='blocker' else 2)
                self.assertTrue((root/'output/failed.json').exists())
                self.assertFalse((root/'output/complete.json').exists())
                if failure!='blocker':
                    self.assertFalse(any(read(root/'output/smoke-profile.json')['qualification'].values()))
                with self.assertRaises(FileExistsError):smoke.run(args)


if __name__ == '__main__':
    unittest.main()
