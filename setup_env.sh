unset LD_LIBRARY_PATH
unset PYTHONPATH

if [ -d "/cds/sw/" ]; then
    # for psana
    source /cds/sw/ds/ana/conda2-v4/inst/etc/profile.d/conda.sh
    export CONDA_ENVS_DIRS=/cds/sw/ds/ana/conda2/inst/envs/
    export DIR_PSDM=/cds/group/psdm
    export SIT_PSDM_DATA=/cds/data/psdm
    export SUBMODULEDIR=/cds/sw/ds/ana/conda2/rel/lcls2_submodules_01152026

    osrel=`uname -r`
    case $osrel in
        *el9*) conda activate daq_20250402_r9;;
        *)     conda activate daq_20250402;;
    esac
elif [ -d "/sdf/group/lcls/" ]; then
    # for s3df
    source /sdf/group/lcls/ds/ana/sw/conda2-v4/inst/etc/profile.d/conda.sh
    export CONDA_ENVS_DIRS=/sdf/group/lcls/ds/ana/sw/conda2/inst/envs
    export DIR_PSDM=/sdf/group/lcls/ds/ana/
    export SIT_PSDM_DATA=/sdf/data/lcls/ds/
    conda activate ps_20241122
else
    echo "CONDA area not found"
    exit 1
fi

AUTH_FILE=$DIR_PSDM"/sw/conda2/auth.sh"
if [ -f "$AUTH_FILE" ]; then
    source $AUTH_FILE
else
  echo "$AUTH_FILE file is missing"
fi

export CUDA_ROOT=/usr/local/cuda
if [ -h "$CUDA_ROOT" ]; then
    export PATH=${CUDA_ROOT}/bin${PATH:+:${PATH}}
    #export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    export MANPATH=${CUDA_ROOT}/man${MANPATH:+:${MANPATH}}
fi

RELDIR="$( cd "$( dirname $(readlink -f "${BASH_SOURCE[0]:-${(%):-%x}}") )" && pwd )"
export PATH=$RELDIR/install/bin:${PATH}
pyver=$(python -c "import sys; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))")
export PYTHONPATH=$RELDIR/install/lib/python$pyver/site-packages
# for procmgr
export TESTRELDIR=$RELDIR/install
export PROCMGR_EXPORT=RDMAV_FORK_SAFE=1,RDMAV_HUGEPAGES_SAFE=1  # See fi_verbs man page regarding fork()
export PROCMGR_EXPORT=$PROCMGR_EXPORT,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,PS_PARALLEL='none'

# for daqbatch
export DAQMGR_EXPORT=RDMAV_FORK_SAFE=1,RDMAV_HUGEPAGES_SAFE=1  # See fi_verbs man page regarding fork()
export DAQMGR_EXPORT=$DAQMGR_EXPORT,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,PS_PARALLEL='none'

# cpo: seems that in more recent versions blas is creating many threads
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# cpo: getting intermittent file-locking issue on ffb, so try this
export HDF5_USE_FILE_LOCKING=FALSE
# for libfabric. decreases performance a little, but allows forking
export RDMAV_FORK_SAFE=1
export RDMAV_HUGEPAGES_SAFE=1

# needed by JupyterLab
export JUPYTERLAB_WORKSPACES_DIR=${HOME}

# cpo: workaround a qt bug which may no longer be there (dec 5, 2022)
if [ ! -d /usr/share/X11/xkb ]; then
    export QT_XKB_CONFIG_ROOT=${CONDA_PREFIX}/lib
fi

# needed by Ric to get correct libfabric man pages
export MANPATH=$CONDA_PREFIX/share/man${MANPATH:+:${MANPATH}}

# For GPU builds, until a better solution is found
export CUSZ_DIR=/cds/home/c/claus/git/cuSZ/install/lib64/cmake/CUSZ
export cuSZp_DIR=/cds/home/c/claus/git/cuSZp/install/cmake
