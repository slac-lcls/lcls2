"""Internal MPI placement probe; invoked only through the native adapter launcher."""
import os
from pathlib import Path
import sys

from native_contract import artifact, digest, file_hash, read, require, runtime_evidence, write, check_mpi
from native_worker import gpu_identity, physical_cores, step_gpu_environment
from prepare_runtime import check_cpu_imports


def observe_rank(folder, rank):
    affinity = sorted(os.sched_getaffinity(0))
    topology, errors = {}, {}
    for cpu in affinity:
        try:
            topology[str(cpu)] = physical_cores([cpu])[0]
        except (OSError, ValueError) as error:
            errors[str(cpu)] = repr(error)
    status = Path('/proc/self/status').read_text()
    row = dict(rank=rank, node=os.uname().nodename, affinity=affinity,
               topology=topology, topology_errors=errors,
               cpus_allowed=[line for line in status.splitlines() if line.startswith('Cpus_allowed')],
               scheduler={k:os.environ.get(k) for k in ('SLURM_JOB_ID','SLURM_STEP_ID',
                   'SLURM_LOCALID','SLURM_CPUS_PER_TASK','SLURM_JOB_GPUS','SLURM_STEP_GPUS','CUDA_VISIBLE_DEVICES')})
    write(folder/('rank-%d.observed.json'%rank), row)
    return row


def observed_placement(folder, rank, site, case):
    observed = observe_rank(folder, rank)
    require(rank == int(os.environ['SLURM_LOCALID']), 'single-node rank mismatch: '+repr(observed))
    require(os.environ['SLURM_JOB_ID'] == site['job_id'] and observed['node'] == site['node'], 'allocation mismatch: '+repr(observed))
    expected = site['cpu_affinity'][str(case['bds'])][rank]
    require(observed['affinity'] == expected, 'exact CPU affinity mismatch: expected=%r observed=%r'%(expected,observed))
    return observed['affinity'], physical_cores(observed['affinity'])


def probe(request_path, folder, stage):
    require(stage == 'placement', 'placement probe only')
    request, folder = read(request_path), Path(folder)
    require(Path(__file__).resolve() == artifact(request, 'native_worker'), 'unfrozen placement probe')
    source = Path(__file__).resolve().parent
    require(all(file_hash(source/name) == sha for name,sha in request['sources'].items()), 'probe sources changed')
    site, runtime, case = request['site_manifest'], request['runtime'], request['case']
    rank = int(os.environ['SLURM_PROCID'])
    affinity, cores = observed_placement(folder, rank, site, case)
    if rank == 0:
        write(folder/'step.json', dict(job=site['job_id'], node=site['node'], step=os.environ['SLURM_STEP_ID']))
    is_gpu = case['gpus'] == 1 and rank >= 2
    gpu_environment = None
    inherited_visibility = os.environ.get('CUDA_VISIBLE_DEVICES')
    if not is_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from mpi4py import MPI
    from psana import DataSource
    comm = MPI.COMM_WORLD
    require(comm.rank == rank and comm.size == case['bds']+2, 'MPI topology mismatch')
    check_mpi(site, MPI.Get_library_version())
    check_cpu_imports()
    identity = None
    if is_gpu:
        import cupy as cp
        try:
            identity = gpu_identity()
            write(folder/('rank-%d.gpu-observed.json'%rank), dict(identity=identity))
        except BaseException as error:
            write(folder/('rank-%d.gpu-error.json'%rank), dict(error=repr(error)))
            raise
        gpu_environment = step_gpu_environment(os.environ, site)
        require(identity['uuid'] == site['gpu_uuid'] and 'A100' in identity['model'],
                'wrong shared GPU: expected=%s observed=%r'%(site['gpu_uuid'],identity))
        require(cp.__version__ == runtime['cupy_version'] and
                cp.cuda.runtime.runtimeGetVersion() == runtime['cuda_runtime_version'] and
                cp.cuda.runtime.driverGetVersion() == runtime['cuda_driver_version'], 'GPU runtime differs')
        require(cp.arange(8, dtype=cp.int32).sum().item() == 28, 'GPU probe failed')
        cp.cuda.runtime.deviceSynchronize()
    provenance = runtime_evidence(runtime)
    if not is_gpu:
        check_cpu_imports()
    comm.Barrier()
    require(comm.allreduce(rank) == sum(range(comm.size)), 'MPI collective failed')
    row = dict(rank=rank, job=site['job_id'], node=os.uname().nodename, affinity=affinity,
               physical_cores=cores, gpu=identity, cpu_no_cuda=not is_gpu, gpu_environment=gpu_environment,
               inherited_cuda_visible_devices=inherited_visibility, provenance=provenance,
               environment_keys=sorted(os.environ), environment_sha256=digest(dict(os.environ)))
    write(folder/('rank-%d.json'%rank), row)
    rows = comm.gather(row, root=0)
    if rank == 0:
        write(folder/'complete.json', dict(status='pass', stage=stage, rows=rows))
    comm.Barrier()


if __name__ == '__main__':
    probe(*sys.argv[1:])
