"""Cross-site dispatch and physical-counter failure contracts; no hardware required."""
import copy
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import native_contract as n
import perlmutter_adapter as adapter
import select_site
import campaign
import sdf_rank
import native_worker
from test_native import NativeTests


class RoutingTests(unittest.TestCase):
    def test_cluster_default_and_explicit_remote(self):
        self.assertEqual(select_site.select(environment={}, hostname='sdfiana025')['execution_site'],'sdf')
        self.assertEqual(select_site.select(environment={'NERSC_HOST':'perlmutter'}, hostname='login40')['execution_site'],'perlmutter')
        remote=select_site.select('perlmutter',{},'sdfiana025')
        self.assertTrue(remote['remote_required'])
        self.assertEqual(remote['execution_site'],'perlmutter')
        self.assertFalse(select_site.select('sdf',{},'sdfiana025')['remote_required'])
        self.assertTrue(select_site.select('sdf',{},'unknown')['remote_required'])
        with self.assertRaisesRegex(ValueError,'unknown cluster'):select_site.select(environment={},hostname='unknown')
        with self.assertRaisesRegex(ValueError,'conflicting'):
            select_site.select(environment={'NERSC_HOST':'perlmutter'},hostname='sdfiana025')

    def test_login_cannot_become_compute_by_setting_job_id(self):
        for host in ('sdfiana025','sdflogin001','login40','sdfhopper001'):
            with self.subTest(host=host), self.assertRaises(ValueError):
                n.compute_site({'SLURM_JOB_ID':'123'},host)
        for host,site in (('sdfampere001','sdf'),('nid000001','perlmutter')):
            self.assertEqual(n.compute_site({'SLURM_JOB_ID':'123'},host),site)
            with self.assertRaises(ValueError):n.compute_site({},host)

    def test_sdf_reserves_allocated_gpu_set_but_exposes_one_uuid(self):
        site=dict(schema='sdf-raw-site-v1',mpi_launcher='openmpi-local-v1',job_id='123',node='sdfampere001',
                  gpu_uuid='GPU-11111111-1111-1111-1111-111111111111',job_gpus=['0','1','2','3'],step_gpu_count=4)
        env=dict(SLURM_JOB_GPUS='0,1,2,3',SLURM_STEP_GPUS='0,1,2,3',CUDA_VISIBLE_DEVICES=site['gpu_uuid'])
        self.assertEqual(native_worker.step_gpu_environment(env,site),env)
        argv=adapter.launch_argv(site,4,4,['0xf','0xf0','0xf00','0xf000'],1,'sample')
        self.assertIn('--gpus-per-node=4',argv)
        for key,value in (('SLURM_STEP_GPUS','2'),('SLURM_STEP_GPUS','0,1,2,2'),('CUDA_VISIBLE_DEVICES','0'),('SLURM_JOB_GPUS','1')):
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                native_worker.step_gpu_environment(dict(env,**{key:value}),site)
        self.assertIn('--gpus-per-node=0',adapter.launch_argv(site,4,4,['0xf','0xf0','0xf00','0xf000'],0,'sample'))

    def test_launch_mpi_and_gpu_controls(self):
        for site,plugin in (('sdf','none'),('perlmutter','cray_shasta')):
            config=dict(schema=site+'-raw-site-v1',mpi_launcher='openmpi-local-v1',step_gpu_count=1,job_id='123',node='node')
            argv=adapter.launch_argv(config,4,4,['0xf','0xf0','0xf00','0xf000'],1,'sample')
            self.assertIn('--mpi='+plugin,argv)
            self.assertIn('--gpus-per-node=1',argv)
            self.assertIn('--exclusive',argv)
            self.assertIn('--cpu-bind=mask_cpu:'+('0xffff' if site=='sdf' else '0xf,0xf0,0xf00,0xf000'),argv)
            self.assertIn('--ntasks='+('1' if site=='sdf' else '4'),argv)
        with self.assertRaises(ValueError):adapter.mpi_plugin(dict(schema='sdf-raw-site-v1',mpi_launcher='cray_shasta'))
        n.check_mpi(dict(schema='sdf-raw-site-v1'),'Open MPI v5.0.8')
        with self.assertRaises(ValueError):n.check_mpi(dict(schema='sdf-raw-site-v1'),'MPICH')


class RankBindingTests(unittest.TestCase):
    def test_local_openmpi_rank_mapping_and_cpu_allowance(self):
        site=dict(schema='sdf-raw-site-v1',job_id='123',node='sdfampere001',cpu_affinity={'1':[[0],[1],[2]]},gpu_uuid='GPU-123')
        request=dict(site_manifest=site,case=dict(bds=1,gpus=1))
        env=dict(SLURM_JOB_ID='123',SLURM_STEP_ID='0',SLURM_PROCID='0',SLURM_LOCALID='0',
                 OMPI_COMM_WORLD_RANK='2',OMPI_COMM_WORLD_LOCAL_RANK='2',OMPI_COMM_WORLD_SIZE='3')
        with tempfile.TemporaryDirectory() as temp, patch.object(sdf_rank,'compute_site',return_value='sdf'), \
             patch.object(sdf_rank.os,'uname',return_value=SimpleNamespace(nodename='sdfampere001')), \
             patch.object(sdf_rank.os,'sched_getaffinity',side_effect=[{0,1,2},{2}]), \
             patch.object(sdf_rank.os,'sched_setaffinity') as setter:
            sdf_rank.bind(request,temp,env)
            setter.assert_called_once_with(0,[2])
            self.assertEqual(env['SLURM_PROCID'],'2')
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'GPU-123')
            self.assertEqual(n.read(Path(temp)/'sdf-rank-2-launch.json')['scheduler_launcher_rank'],0)
        for failure in ('rank','size','scheduler','allocation','cpus'):
            bad=dict(env,SLURM_PROCID='0')
            if failure=='rank':bad['OMPI_COMM_WORLD_LOCAL_RANK']='1'
            if failure=='size':bad['OMPI_COMM_WORLD_SIZE']='4'
            if failure=='scheduler':bad['SLURM_PROCID']='1'
            if failure=='allocation':bad['SLURM_JOB_ID']='999'
            with tempfile.TemporaryDirectory() as temp, patch.object(sdf_rank,'compute_site',return_value='sdf'), \
                 patch.object(sdf_rank.os,'uname',return_value=SimpleNamespace(nodename='sdfampere001')), \
                 patch.object(sdf_rank.os,'sched_getaffinity',return_value={0}), \
                 patch.object(sdf_rank.os,'sched_setaffinity') as setter, self.subTest(failure=failure),self.assertRaises(ValueError):
                sdf_rank.bind(request,temp,bad)
            setter.assert_not_called()


class PhysicalNicTests(unittest.TestCase):
    def fixture(self, root):
        net=root/'net';net.mkdir()
        dev=root/'0000:c5:00.0';dev.mkdir()
        interface=net/'enp197s0';interface.mkdir();(interface/'device').symlink_to(dev)
        return net,dev

    def test_snapshots_and_byte_rates(self):
        with tempfile.TemporaryDirectory() as temp:
            root,dev=self.fixture(Path(temp))
            output=SimpleNamespace(stdout='rx_bytes_phy: 100\ntx_bytes_phy: 200\nrx_bytes: 99999\n')
            with patch.object(n.subprocess,'run',return_value=output),patch.object(n.time,'monotonic',return_value=1):
                before=n.physical_nic_snapshot(['enp197s0'],{'enp197s0':str(dev)},root)
            output.stdout='rx_bytes_phy: 1124\ntx_bytes_phy: 712\n'
            with patch.object(n.subprocess,'run',return_value=output),patch.object(n.time,'monotonic',return_value=3):
                after=n.physical_nic_snapshot(['enp197s0'],{'enp197s0':str(dev)},root)
            delta=n.traffic_delta(before,after)
            self.assertEqual((delta['rx_bytes'],delta['tx_bytes']),(1024,512))
            self.assertEqual(delta['interfaces']['enp197s0']['rx_gib_s'],512/(1<<30))
            self.assertIn('rx_bytes_phy',before['interfaces']['enp197s0']['raw'])
            for failure in ('reset','clock','device','missing','unavailable','units','backend','duplicate'):
                bad=copy.deepcopy(after)
                row=bad['interfaces']['enp197s0']
                if failure=='reset':row['counters']['rx_bytes_phy']=99
                if failure=='clock':row['monotonic']=1
                if failure=='device':row['device']='/other'
                if failure=='missing':bad['interfaces']={}
                if failure=='unavailable':bad['unavailable']={'enp197s0':'denied'}
                if failure=='units':bad['units']='words'
                if failure=='backend':bad['backend']='netdev'
                prior=copy.deepcopy(before)
                if failure=='duplicate':
                    bad['interfaces']['alias']=copy.deepcopy(row)
                    prior['interfaces']['alias']=copy.deepcopy(prior['interfaces']['enp197s0'])
                with self.subTest(failure=failure),self.assertRaises(ValueError):n.physical_nic_delta(prior,bad)

    def test_no_virtual_duplicate_missing_or_denied_as_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root,dev=self.fixture(Path(temp))
            (root/'bond0').mkdir()
            (root/'alias').mkdir();(root/'alias/device').symlink_to(dev)
            result=SimpleNamespace(stdout='rx_bytes_phy: 1\ntx_bytes_phy: 2\n')
            with patch.object(n.subprocess,'run',return_value=result):
                for names in ([],['bond0'],['../bad'],['enp197s0','enp197s0'],['enp197s0','alias']):
                    with self.assertRaises(ValueError):n.physical_nic_snapshot(names,root=root)
                for raw in ('rx_bytes: 3\ntx_bytes: 4\n','rx_bytes_phy: -1\ntx_bytes_phy: 4\n','rx_bytes_phy: 1\nrx_bytes_phy: 2\ntx_bytes_phy: 4\n'):
                    result.stdout=raw
                    observed=n.physical_nic_snapshot(['enp197s0'],root=root)
                    self.assertTrue(observed['unavailable'])
                    with self.assertRaises(ValueError):n.physical_nic_delta(observed,observed)
            with patch.object(n.subprocess,'run',side_effect=subprocess.TimeoutExpired('ethtool',5)):
                self.assertTrue(n.physical_nic_snapshot(['enp197s0'],root=root)['unavailable'])


class SdfContractTests(unittest.TestCase):
    setUp,tearDown=NativeTests.setUp,NativeTests.tearDown
    add_artifact=NativeTests.add_artifact
    rows=NativeTests.rows

    def request_for_sdf(self):
        self.config['site']='sdf'
        self.config['dataset']['sites']['sdf']=str(self.root)
        self.config['dataset']['sites']['perlmutter']='/foreign/perlmutter/path'
        (self.root/'mpiexec').write_text('fixture')
        self.site.update(schema='sdf-raw-site-v1',node='sdfampere001',mpi_launcher='openmpi-local-v1',mpiexec=str(self.root/'mpiexec'),mpiexec_sha256=n.file_hash(self.root/'mpiexec'),
                         job_gpus=['0'],step_gpu_count=1,nic_backend='ethtool-physical-v1',nic_interfaces=['enp197s0'],nic_devices={'enp197s0':'/pci/0000:c5:00.0'})
        self.add_artifact('native_site',self.site)
        return campaign.make_request(self.config,{'profile_sha256':n.digest(self.config)},campaign.schedule(self.config,'correctness')[0])

    def test_launch_contains_local_only_mpi_and_frozen_rank_wrapper(self):
        request=self.request_for_sdf()
        wrapper=self.root/'sdf_rank.py';wrapper.write_text('fixture')
        request['artifacts']['sdf_rank']=dict(path=str(wrapper),sha256=n.file_hash(wrapper))
        process=Mock();process.wait.side_effect=[subprocess.TimeoutExpired(['srun'],1),0];process.poll.return_value=None
        folder=self.root/'launch'
        with patch.object(adapter,'matching_steps',return_value=[]), \
             patch.object(adapter,'cancel_step',return_value={'no_active_steps':True}), \
             patch.object(adapter.subprocess,'Popen',return_value=process):
            with self.assertRaises(subprocess.TimeoutExpired):
                adapter.launch(request,self.root/'request.json',folder,'measured',self.site,self.runtime)
        argv=n.read(folder/'launch.json')['argv']
        self.assertIn('--mpi=none',argv)
        self.assertIn('--ntasks=1',argv)
        self.assertIn(str(wrapper),argv)
        self.assertEqual(argv[argv.index('--host')+1],'localhost')
        self.assertEqual(argv[argv.index('plm_ssh_agent')+1],'/usr/bin/false')
        self.assertEqual(argv[argv.index('ras')+1],'^slurm')
        self.assertTrue(n.read(folder/'failure.json')['cleanup']['no_active_steps'])

    def test_sdf_results_require_physical_nic_and_preserve_reporting(self):
        request=self.request_for_sdf()
        rows=self.rows()
        for row in rows:
            row.update(node=self.site['node'],affinity=[row['rank']],gpu=None,
                observed_settings={'batch_size':20},environment=dict(request['contract']['environment'],PS_SMD_N_EVENTS='1000',PS_PARALLEL='mpi'),
                provenance_before={},provenance_after={},setup_seconds=1,first_batch_seconds=2 if row['rank']>=2 else None,loop_seconds=5)
        before=dict(backend='ethtool-physical-v1',units='bytes',unavailable={},interfaces={'enp197s0':dict(
            device='/pci/0000:c5:00.0',monotonic=1,counters=dict(rx_bytes_phy=100,tx_bytes_phy=0))})
        after=copy.deepcopy(before);after['interfaces']['enp197s0'].update(monotonic=2,counters=dict(rx_bytes_phy=2100,tx_bytes_phy=50))
        cache=dict(policy='client-cold',verified=True,measured_bytes=2000,resident_bytes_before=0)
        measured=dict(rows=rows,diagnostic=False,work=dict(events=4,event_sha256=n.digest([100,101,102,103]),requested_bytes=2000),
            cache={'observations':[cache,cache]},nic=dict(rx_bytes=2000,tx_bytes=50),nic_before={},nic_after={},
            physical_nic_before=before,physical_nic_after=after)
        warmup=dict(status='pass',stage='warmup',rows=rows)
        for failure in ('missing','low','wrong-total','unavailable'):
            bad=copy.deepcopy(measured)
            if failure=='missing':bad.pop('physical_nic_before')
            if failure=='low':
                bad['physical_nic_after']['interfaces']['enp197s0']['counters']['rx_bytes_phy']=200
                bad['nic']['rx_bytes']=100
            if failure=='wrong-total':bad['nic']['rx_bytes']=9999
            if failure=='unavailable':bad['physical_nic_after']['unavailable']={'enp197s0':'denied'}
            with self.subTest(failure=failure),self.assertRaises((ValueError,KeyError)):
                adapter.assemble(request,bad,warmup,{'measured':{'seconds':6}},7,self.root,self.site)
        result=adapter.assemble(request,measured,warmup,{'measured':{'seconds':6}},7,self.root,self.site)
        campaign.validate_result(self.config,request,result,self.root)
        resources=n.read(self.root/'resources.json')
        self.assertEqual(result['nic'],dict(rx_bytes=2000,tx_bytes=50))
        self.assertEqual(resources['physical_nic_delta']['backend'],'ethtool-physical-v1')

    def test_site_specific_input_and_evidence_required(self):
        request=self.request_for_sdf()
        with patch.dict(os.environ,{'SLURM_JOB_ID':'123','SLURM_JOB_GPUS':'0'},clear=True),patch.object(n.os,'uname',return_value=SimpleNamespace(nodename='sdfampere001')):
            self.assertEqual(n.load_contract(request)[2]['root'],str(self.root))
            for key,value in (('schema','perlmutter-raw-site-v1'),('mpi_launcher','cray_shasta'),('nic_devices',{}),('nic_backend','netdev')):
                original=copy.deepcopy(self.site)
                self.site[key]=value;self.add_artifact('native_site',self.site)
                request['artifacts']=self.config['artifacts']
                with self.subTest(key=key),self.assertRaises(ValueError):n.load_contract(request)
                self.site=original
            self.add_artifact('native_site',self.site)
            request['artifacts']=self.config['artifacts']
            request['contract']['dataset']['sites']['sdf']='/wrong'
            request['contract_sha256']=n.digest(request['contract'])
            with self.assertRaisesRegex(ValueError,'input root'):n.load_contract(request)


del NativeTests
if __name__=='__main__':unittest.main()
