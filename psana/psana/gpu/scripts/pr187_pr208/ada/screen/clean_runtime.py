"""Fresh runtime environment per sample: no inherited Python/compiler/library paths."""
from pathlib import Path
DEP=Path('/sdf/scratch/users/m/monarin/gpu-validation/kvikio-2610-validation-20261015-r1')
BASE=Path('/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/xpp_drp_gpu_311')
def runtime_environment(root,variant,parent):
 build=root/variant
 env={k:v for k,v in parent.items() if k.startswith('SLURM_') or k in ('HOME','USER','LOGNAME','LANG','CUDA_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL')}
 env.update(BENCH_ALIGNMENT_SHIFT='0',PSANA_GPU_SHARE_CONSTANTS='1',PATH=f'{DEP}/venv/bin:{BASE}/bin:/opt/slurm/slurm-curr/bin:/usr/bin:/bin',
   LD_LIBRARY_PATH=f'{build}/install/lib:{BASE}/lib',
   PYTHONPATH=':'.join(str(build/x) for x in ('install/lib/python3.11/site-packages','scripts/jf_scaling','scripts/feespec_bulk_benchmark','scripts/stage1_regression')),
   PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
   TMPDIR=str(build/'tmp'),CUPY_CACHE_DIR=str(build/'cupy-cache'),
   OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PS_PARALLEL='mpi',PS_EB_NODES='1',PS_SRV_NODES='0',PS_SMD_N_EVENTS='1000',
   OMPI_MCA_pml='ob1',OMPI_MCA_btl='^smcuda',OMPI_MCA_rmaps_base_oversubscribe='1',
   KVIKIO_COMPAT_MODE='ON',KVIKIO_NTHREADS='8',KVIKIO_TASK_SIZE='1048576',
   KVIKIO_EXPECTED_ROOT=str(DEP/'venv/lib/python3.11/site-packages'),
   BENCH_INSTALL=str(build/'install'),BENCH_PYTHON=str(build/'install/lib/python3.11/site-packages'),
   LCLS_CALIB_HTTP='https://pswww.slac.stanford.edu/ws')
 return env
