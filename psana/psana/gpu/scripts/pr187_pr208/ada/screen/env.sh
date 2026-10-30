unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH CONDA_PREFIX VIRTUAL_ENV CUDA_PATH CUDA_HOME CUDA_ROOT
export PATH=/sdf/scratch/users/m/monarin/gpu-validation/kvikio-2610-validation-20261015-r1/venv/bin:/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/xpp_drp_gpu_311/bin:/opt/slurm/slurm-curr/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/xpp_drp_gpu_311/lib
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset PYTHONPATH LD_PRELOAD
export TMPDIR=/sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-performance-ada016-20261027-r2/screen/tmp CUPY_CACHE_DIR=/sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-performance-ada016-20261027-r2/screen/cupy-cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export KVIKIO_COMPAT_MODE=ON KVIKIO_NTHREADS=8 KVIKIO_TASK_SIZE=1048576
export PS_PARALLEL=mpi PS_EB_NODES=1 PS_SRV_NODES=0 PS_SMD_N_EVENTS=1000
export OMPI_MCA_btl=^smcuda OMPI_MCA_rmaps_base_oversubscribe=1
export LCLS_CALIB_HTTP=https://pswww.slac.stanford.edu/ws
unset SIT_PSDM_OFFSITE PSANA_GPU_MEM_DEBUG
export KVIKIO_EXPECTED_ROOT=/sdf/scratch/users/m/monarin/gpu-validation/kvikio-2610-validation-20261015-r1/venv/lib/python3.11/site-packages
export PYTHONPATH=/sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-performance-ada016-20261027-r2/screen/A/scripts/jf_scaling:/sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-performance-ada016-20261027-r2/screen/A/scripts/feespec_bulk_benchmark:/sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-performance-ada016-20261027-r2/screen/A/scripts/stage1_regression
unset BENCH_PYTHON
cd /sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-performance-ada016-20261027-r2/screen/work

export OMPI_MCA_pml=ob1
