"""Offline site preparation tests; hardware qualification is separate."""
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import prepare_site as site
import site_probe
from native_contract import file_hash, read, write


class SiteTests(unittest.TestCase):
    def test_smt_dedup_and_prefix(self):
        topology = {c:[0,c%16] for c in range(32)}
        p = site.placements(topology,4)
        self.assertEqual(p['2'],[list(range(i,i+4)) for i in range(0,16,4)])
        self.assertEqual(p['1'],p['2'][:3])

    def test_insufficient_physical_cores(self):
        with self.assertRaisesRegex(ValueError,'insufficient'):
            site.placements({c:[0,c%8] for c in range(16)},4)

    def test_invalid_core_count(self):
        for count in (0,33,True,1.5):
            with self.assertRaises(ValueError): site.placements({},count)

    def test_timeout_boundary(self):
        site.timeouts(389,900)
        for step,controller in ((390,900),(0,900),(3600,8000),(180,480)):
            with self.assertRaises(ValueError): site.timeouts(step,controller)

    def test_inventory(self):
        gpu='GPU-12345678-1234-1234-1234-123456789abc'
        self.assertEqual(site.gpu_inventory('0, '+gpu+', NVIDIA A100, 0000:01:00.0')[0]['uuid'],gpu)
        for value in ('', '0, not-a-uuid, NVIDIA A100, pci', '0, '+gpu+', NVIDIA H100, pci'):
            with self.assertRaises(ValueError): site.gpu_inventory(value)

    def test_physical_nics_only_and_unique(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); net=root/'net';net.mkdir()
            (net/'lo').mkdir();(net/'bond0').mkdir()
            driver=root/'cxi';driver.mkdir()
            pci=root/'0000:41:00.0';pci.mkdir();(pci/'driver').symlink_to(driver)
            hsn=net/'hsn0';hsn.mkdir();(hsn/'device').symlink_to(pci)
            (hsn/'type').write_text('1');(hsn/'operstate').write_text('up')
            (hsn/'statistics').mkdir()
            for direction in ('rx','tx'):(hsn/'statistics'/(direction+'_bytes')).write_text('42')
            rows=site.nic_inventory(net)
            self.assertEqual([r['interface'] for r in rows],['hsn0'])
            (net/'hsn1').symlink_to(hsn)
            with self.assertRaisesRegex(ValueError,'duplicate'):site.nic_inventory(net)

    def test_virtual_hsn_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'hsn0').mkdir()
            with self.assertRaisesRegex(ValueError,'physical'):site.nic_inventory(root)

    def fixture(self):
        topology={c:[0,c] for c in range(16)}
        manifest=dict(node='nid000001',job_id='123',gpu_uuid='GPU-12345678-1234-1234-1234-123456789abc',cpu_affinity=site.placements(topology,4))
        return topology,manifest

    def rows(self, manifest, bds, gpus):
        return [dict(rank=r,node=manifest['node'],job=manifest['job_id'],affinity=cpus,
                     physical_cores=[[0,c] for c in cpus],cpu_no_cuda=not(gpus and r>=2),
                     gpu=dict(uuid=manifest['gpu_uuid'],model='A100') if gpus and r>=2 else None)
                for r,cpus in enumerate(manifest['cpu_affinity'][str(bds)])]

    def test_observed_placement_rejected(self):
        topology,manifest=self.fixture()
        rows=self.rows(manifest,2,1)
        site.validate_rows(rows,manifest,2,1,topology)
        for key,value in (('affinity',[99]),('physical_cores',[[9,9]]),('gpu',None),('node','nid999999')):
            bad=copy.deepcopy(rows);bad[2][key]=value
            with self.assertRaises(ValueError):site.validate_rows(bad,manifest,2,1,topology)
        bad=copy.deepcopy(rows);bad[0]['cpu_no_cuda']=False
        with self.assertRaises(ValueError):site.validate_rows(bad,manifest,2,1,topology)

    def run_capture(self, root, failure=None, mode=None):
        runtime=root/'runtime.json';write(runtime,dict(schema='psana-native-runtime-v1',path_policy='isolated-install-v1'))
        args=SimpleNamespace(runtime=runtime,runtime_sha256=file_hash(runtime),output=root/'site.json',
                             cores_per_rank=4,step_timeout=180,controller_timeout=900)
        if mode == 'existing-output':write(args.output,dict(original=True))
        if mode == 'existing-folder':(root/'site.json.capture').mkdir()
        environment=dict(SLURM_JOB_ID='123',SLURM_STEP_ID='batch',SLURM_JOB_NUM_NODES='1',SLURM_JOB_GPUS='0')
        if mode == 'wrong-step':environment['SLURM_STEP_ID']='0'
        if mode == 'multiple-nodes':environment['SLURM_JOB_NUM_NODES']='2'
        if mode == 'missing-gpu':environment['SLURM_JOB_GPUS']='1'
        changed=[False]
        def hashed(path):
            if mode == 'source-change' and changed[0] and Path(path).name == 'site_probe.py':return 'changed'
            return file_hash(path)
        gpu='GPU-12345678-1234-1234-1234-123456789abc'
        def command(argv):
            return 'JobId=123 NumNodes=1 ' if argv[0]=='scontrol' else '0, '+gpu+', A100, 0000:01:00.0'
        def launch(request,path,folder,stage,manifest,runtime):
            folder.mkdir();write(folder/'launch.json',dict(stage=stage))
            if failure:raise RuntimeError(failure)
            changed[0]=True
            if mode == 'runtime-change':args.runtime.write_text(args.runtime.read_text()+'\n')
            return dict(rows=self.rows(manifest,request['case']['bds'],request['case']['gpus'])),dict(status='pass')
        snapshot=dict(interfaces={'hsn0':dict(rx=1,tx=2)},unavailable={})
        with patch.dict(os.environ,environment), patch.object(site,'file_hash',side_effect=hashed), \
             patch.object(site.os,'uname',return_value=SimpleNamespace(nodename='nid000001')), \
             patch.object(site.os,'sched_getaffinity',return_value=set(range(64))), \
             patch.object(site,'physical_cores',side_effect=lambda cpus:[[0,c] for c in cpus]), \
             patch.object(site,'command_output',side_effect=command), \
             patch.object(site,'nic_inventory',return_value=[dict(interface='hsn0')]), \
             patch.object(site,'nic_snapshot',side_effect=[snapshot,dict(interfaces={},unavailable={'hsn0':'missing'}) if mode == 'missing-nic' else snapshot]),patch.object(site,'launch',side_effect=launch):
            site.capture(args)
        return args

    def test_publish_after_four_probes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args=self.run_capture(root)
            self.assertEqual(read(args.output)['job_id'],'123')
            self.assertEqual(read(args.output)['step_cpus_per_task'],16)
            evidence=read(root/'site.json.capture'/'validation.json')
            self.assertEqual(len(evidence['probes']),4)
            self.assertEqual(evidence['nic_delta'],dict(rx_bytes=0,tx_bytes=0))
            index=root/'site.json.capture'/'index.json'
            self.assertEqual(read(args.output)['capture_index']['sha256'],file_hash(index))
            self.assertEqual(read(root/'site.json.capture'/'site-draft.json')['schema'],'perlmutter-raw-site-draft-v1')

    def test_failed_launch_never_publishes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.assertRaisesRegex(RuntimeError,'bad binding'):self.run_capture(root,'bad binding')
            self.assertFalse((root/'site.json').exists())
            self.assertIn('bad binding',read(root/'site.json.capture'/'failed.json')['error'])
            self.assertTrue((root/'site.json.capture'/'site-draft.json').exists())

    def test_capture_guards(self):
        for mode in ('wrong-step','multiple-nodes','missing-gpu','existing-output','existing-folder',
                     'runtime-change','source-change','missing-nic'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                with self.assertRaises((ValueError,FileExistsError)):self.run_capture(root,mode=mode)
                if mode == 'existing-output':self.assertEqual(read(root/'site.json'),dict(original=True))
                else:self.assertFalse((root/'site.json').exists())
                if mode in ('missing-gpu','runtime-change','source-change','missing-nic'):
                    self.assertTrue((root/'site.json.capture'/'failed.json').exists())

    def test_probe_failure_preserves_actual_smt_observation(self):
        _,manifest=self.fixture()
        with tempfile.TemporaryDirectory() as temp, \
             patch.dict(os.environ,dict(SLURM_LOCALID='0',SLURM_JOB_ID='123')), \
             patch.object(site_probe.os,'uname',return_value=SimpleNamespace(nodename='nid000001')), \
             patch.object(site_probe.os,'sched_getaffinity',return_value={0,1,128,129}), \
             patch.object(site_probe,'physical_cores',side_effect=lambda cpus:[[0,c%128] for c in cpus]):
            folder=Path(temp)
            with self.assertRaisesRegex(ValueError,'expected=.*observed='):
                site_probe.observed_placement(folder,0,manifest,dict(bds=1))
            observed=read(folder/'rank-0.observed.json')
            self.assertEqual(observed['affinity'],[0,1,128,129])
            self.assertEqual(observed['topology']['0'],observed['topology']['128'])
            self.assertTrue(observed['cpus_allowed'])


if __name__=='__main__':unittest.main()
