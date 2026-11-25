"""Slurm step arguments: ranked placement comes exclusively from a hostfile."""
import math
from topology import rank_hosts


def step_command(bundle,attempt,hosts,program,args,timeout,ranked=False):
    count=len(rank_hosts(hosts)) if ranked else len(hosts)
    # Slurm25.11 rejects explicit --nodes with arbitrary distribution.
    # A --nodelist would override SLURM_HOSTFILE and lose repeated rank hosts.
    placement=[] if ranked else ['--nodes='+str(len(hosts)),'--nodelist='+','.join(hosts)]
    return ['srun','--overlap','--mpi=cray_shasta',*placement,
            '--ntasks='+str(count),'--distribution='+('arbitrary' if ranked else 'block'),
            '--cpus-per-task='+('20' if ranked else '128'),
            '--gpus-per-node=4','--gpu-bind=none','--cpu-bind=none',
            '--kill-on-bad-exit=1','--time='+str(math.ceil(timeout/60)),
            'numactl','--interleave=all','/usr/bin/python3',str(bundle/'clean_exec.py'),
            str(attempt),'python',str(bundle/program),*map(str,args)]


def step_environment(parent,hostfile=None):
    env=dict(parent)
    for key in ('SLURM_HOSTFILE','SLURM_NTASKS_PER_NODE','SLURM_TASKS_PER_NODE','SLURM_DISTRIBUTION'):
        env.pop(key,None)
    if hostfile:
        # Do not inherit the outer coordinator step's node/task layout.
        for key in ('SLURM_NNODES','SLURM_JOB_NUM_NODES','SLURM_NODELIST','SLURM_NTASKS','SLURM_NPROCS'):
            env.pop(key,None)
        env['SLURM_HOSTFILE']=str(hostfile)
    return env
