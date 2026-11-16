import json
import os
from pathlib import Path
import time


def rank_environment():
    rank = int(os.environ['SLURM_PROCID'])
    local = int(os.environ['SLURM_LOCALID'])
    if rank != local or rank < 0:
        raise RuntimeError('Expected single-node Slurm rank placement')
    return rank


def physical_cores(cpus):
    result = set()
    for cpu in cpus:
        p = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        result.add(((p/'physical_package_id').read_text().strip(), (p/'core_id').read_text().strip()))
    return sorted(result)


def validate_rows(rows, mode):
    if sorted(row['rank'] for row in rows) != list(range(len(rows))):
        raise RuntimeError('Missing/duplicate MPI ranks')
    used = set()
    for row in rows:
        expected = 'smd0' if row['rank'] == 0 else ('eb' if row['rank'] == 1 else 'bd')
        if row['role'] != expected or row['local_rank'] != row['rank']:
            raise RuntimeError('Unexpected rank role/local placement')
        cores = set(tuple(x) for x in row['physical_cores'])
        if not cores or used & cores:
            raise RuntimeError('Empty or overlapping rank physical-core bindings')
        used |= cores
        if (mode == 'cpu' or row['rank'] < 2) and row['gpu_uuid'] is not None:
            raise RuntimeError('GPU on CPU role')
        if 'MPICH' not in row['mpi_library']:
            raise RuntimeError('Expected Cray MPICH')
    uuids = {row['gpu_uuid'] for row in rows if row['rank'] >= 2}
    if mode == 'gpu' and (len(uuids) != 1 or None in uuids):
        raise RuntimeError('BDs do not share one physical GPU')


def progress(args, rank, timestamps, started):
    if len(timestamps) % 100 == 0:
        row = dict(rank=rank, events=len(timestamps), seconds=time.perf_counter()-started,
                   unix_time=time.time(), timestamp=timestamps[-1])
        with Path(args.output).with_name(f'progress-rank-{rank}.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')


def runtime_provenance(gpu):
    """Observe effective compute-node runtime without importing CuPy on CPU ranks."""
    from mpi4py import MPI
    mapped=set()
    markers=('libmpi','libkvikio','libcudart','libcuda','libnvrtc','libcufile')
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)==6 and fields[5].startswith('/') and any(x in Path(fields[5]).name for x in markers):
            mapped.add(str(Path(fields[5]).resolve()))
    expected_mpi='/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/lib/libmpi_gnu_123.so.12.0.0'
    mpi_paths=[p for p in mapped if Path(p).name.startswith('libmpi')]
    if mpi_paths != [expected_mpi]:
        raise RuntimeError(f'Unexpected MPI mappings: {mpi_paths}')
    result=dict(mpi=MPI.Get_library_version(),mapped_native_libraries=sorted(mapped),gpu_initialized=gpu is not None)
    if gpu is not None:
        import kvikio, kvikio.defaults
        conda=Path('/global/homes/m/monarin/.conda-envs/psana-build').resolve()
        kvikio_so=conda/'lib/python3.9/site-packages/kvikio/_lib/libkvikio.cpython-39-x86_64-linux-gnu.so'
        cuda_root=Path('/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9').resolve()
        cudart_bundled=conda/'lib/python3.9/site-packages/kvikio_cu12.libs/libcudart-16e8815a.so.12.5.82'
        for name in mapped:
            path=Path(name)
            if path.name.startswith('libkvikio') and path != kvikio_so:
                raise RuntimeError(f'Unexpected KvikIO library: {path}')
            if path.name.startswith('libcudart') and path != cudart_bundled and cuda_root not in path.parents:
                raise RuntimeError(f'Unexpected cudart library: {path}')
            if path.name.startswith('libnvrtc') and cuda_root not in path.parents:
                raise RuntimeError(f'Unexpected NVRTC library: {path}')
        compat=kvikio.defaults.compat_mode()
        if compat is not True: raise RuntimeError('Effective KvikIO compatibility mode is not enabled')
        if str(kvikio_so) not in mapped or not any('libcudart' in Path(p).name for p in mapped):
            raise RuntimeError('Expected GPU runtime libraries are not mapped')
        result.update(cupy_version=gpu.__version__,kvikio_version=kvikio.__version__,compat_mode=compat,
                      cuda_runtime_version=gpu.cuda.runtime.runtimeGetVersion(),cuda_driver_version=gpu.cuda.runtime.driverGetVersion(),
                      cupy_module=str(Path(gpu.__file__).resolve()),kvikio_module=str(Path(kvikio.__file__).resolve()))
    return result
