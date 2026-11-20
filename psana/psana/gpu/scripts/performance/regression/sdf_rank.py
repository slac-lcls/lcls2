"""Frozen SDF Open MPI rank binding inside one Slurm step; exec the pinned worker."""
import os
from pathlib import Path
import sys
from native_contract import artifact, compute_site, read, require, write


def bind(request, folder, environment=None):
    env = os.environ if environment is None else environment
    site = request.get('site_manifest')
    if site is None:
        site = request.get('site') if isinstance(request.get('site'), dict) else read(artifact(request,'native_site'))
    require(site['schema']=='sdf-raw-site-v1' and compute_site()=='sdf', 'SDF rank wrapper only')
    require(env.get('SLURM_JOB_ID')==site['job_id'] and os.uname().nodename==site['node'], 'rank allocation differs')
    require(env.get('SLURM_PROCID')==env.get('SLURM_LOCALID')=='0', 'expected one Slurm launcher task')
    rank, local, size = [int(env[k]) for k in ('OMPI_COMM_WORLD_RANK','OMPI_COMM_WORLD_LOCAL_RANK','OMPI_COMM_WORLD_SIZE')]
    require(rank==local and size==request['case']['bds']+2 and 0<=rank<size, 'single-node Open MPI topology differs')
    cpus=site['cpu_affinity'][str(request['case']['bds'])][rank]
    before=sorted(os.sched_getaffinity(0))
    require(set(cpus)<=set(before), 'rank cores outside scheduler allowance')
    os.sched_setaffinity(0,cpus)
    require(sorted(os.sched_getaffinity(0))==sorted(cpus), 'rank affinity did not apply')
    inherited_gpu_visibility=env.get('CUDA_VISIBLE_DEVICES')
    if request['case'].get('gpus') == 1 and rank >= 2:
        env['CUDA_VISIBLE_DEVICES']=site['gpu_uuid']
    write(Path(folder)/('sdf-rank-%d-launch.json'%rank),dict(rank=rank,local_rank=local,size=size,
          job=site['job_id'],step=env['SLURM_STEP_ID'],scheduler_launcher_rank=0,
          affinity_before=before,affinity_after=sorted(cpus),
          inherited_cuda_visible_devices=inherited_gpu_visibility, effective_cuda_visible_devices=env.get('CUDA_VISIBLE_DEVICES'),source='Open MPI ranks within a single Slurm step'))
    # Workers use these rank variables on both sites. Preserve provenance above;
    # the later MPI.COMM_WORLD check independently confirms the mapped identity.
    env['SLURM_PROCID']=str(rank)
    env['SLURM_LOCALID']=str(local)


def main():
    request_path,folder,stage=sys.argv[1:]
    request=read(request_path)
    require(Path(__file__).resolve()==artifact(request,'sdf_rank'),'unfrozen SDF rank wrapper')
    worker=artifact(request,'native_worker')
    bind(request,folder)
    os.execv(sys.executable,[sys.executable,'-B','-s',str(worker),request_path,folder,stage])


if __name__=='__main__':main()
