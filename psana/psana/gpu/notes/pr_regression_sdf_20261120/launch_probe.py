"""Bounded SDF launch/GPU/NIC check; no psana workload or qualification flags.

Copy beside the frozen regression modules for execution. No production dataset
is touched. A 32 MiB private control file tests physical NIC/cache observation.
"""
import os
from pathlib import Path
import sys
import shutil
import time
from native_contract import (compute_site, read, write, require, digest, file_hash,
                             physical_nic_snapshot, physical_nic_delta, cold_cache, check_mpi)
from native_worker import physical_cores, gpu_identity, step_gpu_environment
from prepare_site import placements, gpu_inventory
from perlmutter_adapter import launch, command_output


def worker(request_path, folder, stage):
    request, folder=read(request_path),Path(folder)
    site,case=request['site_manifest'],request['case']
    rank=int(os.environ['SLURM_PROCID'])
    require(compute_site()=='sdf' and rank==int(os.environ['SLURM_LOCALID']), 'wrong site/rank')
    require(sorted(os.sched_getaffinity(0))==site['cpu_affinity'][str(case['bds'])][rank], 'affinity mismatch')
    if rank==0:write(folder/'step.json',dict(job=os.environ['SLURM_JOB_ID'],step=os.environ['SLURM_STEP_ID'],node=os.uname().nodename))
    gpu=case['gpus'] and rank>=2
    if not gpu:os.environ['CUDA_VISIBLE_DEVICES']=''
    from mpi4py import MPI
    check_mpi(site,MPI.Get_library_version())
    require(MPI.COMM_WORLD.rank==rank and MPI.COMM_WORLD.size==case['bds']+2,'MPI topology mismatch')
    identity=None
    if gpu:
        import cupy as cp
        step_gpu_environment(os.environ,site)
        identity=gpu_identity()
        write(folder/('rank-%d-gpu-observed.json'%rank),dict(expected=site['gpu_uuid'],actual=identity,visibility=os.environ.get('CUDA_VISIBLE_DEVICES')))
        require(identity['uuid']==site['gpu_uuid'],'GPU UUID differs: '+repr(identity))
        require(cp.arange(16,dtype=cp.int32).sum().item()==120,'GPU sum differs')
        cp.cuda.runtime.deviceSynchronize()
    require(MPI.COMM_WORLD.allreduce(rank)==sum(range(case['bds']+2)),'MPI collective failed')
    row=dict(rank=rank,affinity=sorted(os.sched_getaffinity(0)),physical_cores=physical_cores(sorted(os.sched_getaffinity(0))),gpu=identity,
             mpi=MPI.Get_library_version(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    write(folder/('rank-%d.json'%rank),row)
    rows=MPI.COMM_WORLD.gather(row,root=0)
    if rank==0:write(folder/'complete.json',dict(status='pass',stage=stage,rows=rows))
    MPI.COMM_WORLD.Barrier()


def driver(output):
    require(compute_site()=='sdf','SDF compute only')
    output=Path(output);output.mkdir(mode=0o700)
    source=Path(__file__).resolve().parent
    write(output/'source-hashes.json',{p.name:file_hash(p) for p in sorted(source.glob('*.py'))})
    (output/'allocation.txt').write_text(command_output(['scontrol','show','job','-d',os.environ['SLURM_JOB_ID']]))
    devices=gpu_inventory(command_output(['nvidia-smi','--query-gpu=index,uuid,name,pci.bus_id','--format=csv,noheader']))
    allocated=os.environ['SLURM_JOB_GPUS'].split(',')
    eligible=[g for g in devices if str(g['index']) in allocated or g['uuid'] in allocated]
    require(eligible,'no allocated GPU')
    nics={}
    for path in sorted(Path('/sys/class/net').iterdir()):
        if (path/'device').exists():
            snap=physical_nic_snapshot([path.name])
            if not snap['unavailable']:nics.update(snap['interfaces'])
    require(nics,'no physical byte counters')
    site=dict(schema='sdf-raw-site-v1',job_id=os.environ['SLURM_JOB_ID'],node=os.uname().nodename,
              gpu_uuid=eligible[0]['uuid'],cpu_affinity=placements({c:physical_cores([c])[0] for c in os.sched_getaffinity(0)},4),
              step_cpus_per_task=4,step_timeout_seconds=90,job_gpus=allocated,step_gpu_count=len(allocated),mpi_launcher='openmpi-local-v1',
              mpiexec=str(Path(shutil.which('mpiexec')).resolve()),mpiexec_sha256=file_hash(Path(shutil.which('mpiexec')).resolve()),
              nic_interfaces=list(nics),nic_devices={k:v['device'] for k,v in nics.items()},nic_backend='ethtool-physical-v1')
    write(output/'site.json',site)
    runtime=dict(python=sys.executable,pythonpath='',environment={k:os.environ[k] for k in
                  ('PATH','LD_LIBRARY_PATH','CUDA_HOME','CUDA_PATH','CONDA_PREFIX','LCLS_CALIB_HTTP')})
    results=[]
    for bds,gpus in ((1,0),(2,0),(1,1),(2,1)):
        name='bd%d-gpu%d'%(bds,gpus)
        request=dict(profile_sha256=digest(site),sample_id=name,case=dict(bds=bds,gpus=gpus),
                     site_manifest=site,contract=dict(environment={}),
                     artifacts=dict(native_worker=dict(path=str(source/'launch_probe.py'),sha256=file_hash(source/'launch_probe.py')),
                                    sdf_rank=dict(path=str(source/'sdf_rank.py'),sha256=file_hash(source/'sdf_rank.py'))))
        path=output/(name+'.json');write(path,request)
        result,scheduler=launch(request,path,output/name,'placement',site,runtime)
        results.append(dict(case=name,rows=result['rows'],scheduler=scheduler))
    data=output/'private-control.bin'
    with data.open('xb') as stream:
        for i in range(32):stream.write(bytes([i])*(1<<20))
        stream.flush();os.fsync(stream.fileno())
    data.chmod(0o400)
    manifest=dict(files=[dict(bigdata=dict(path=str(data),ranges=[[0,32<<20]],range_bytes=32<<20,size=32<<20))])
    cache=cold_cache(manifest,output/'control-cache')
    before=physical_nic_snapshot(site['nic_interfaces'],site['nic_devices'])
    with data.open('rb',buffering=0) as stream:
        payload=stream.read()
    require(len(payload)==32<<20,'control short read')
    time.sleep(.25)
    after=physical_nic_snapshot(site['nic_interfaces'],site['nic_devices'])
    traffic=physical_nic_delta(before,after)
    write(output/'nic-control.json',dict(before=before,after=after,delta=traffic,cache=cache,
          bytes=len(payload),rx_per_selected_byte=traffic['rx_bytes']/len(payload)))
    require(traffic['rx_bytes']>=.9*len(payload),'NIC does not cover private control read')
    write(output/'complete.json',dict(status='pass',cases=results,nic=traffic,
          scope='launch, CPU/GPU assignment, physical NIC and private 32 MiB cache control only; no run51 or performance acceptance'))


if __name__=='__main__':
    if len(sys.argv)==4:
        try:worker(*sys.argv[1:])
        except BaseException:
            import traceback
            traceback.print_exc()
            from mpi4py import MPI
            MPI.COMM_WORLD.Abort(1)
    else:driver(sys.argv[1])
