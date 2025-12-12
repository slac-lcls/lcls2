unset LD_LIBRARY_PATH
unset PYTHONPATH
unset DAQ_CONDA_PREFIX
unset DAQ_CONDA_DEFAULT_ENV
unset DAQ_TESTRELDIR
unset AMI_CONDA_PREFIX
unset AMI_CONDA_DEFAULT_ENV
unset AMI_TESTRELDIR

if [ -d "/cds/sw/" ]; then
    # for psana
    source /cds/sw/ds/ana/conda2-v4/inst/etc/profile.d/conda.sh
    export CONDA_ENVS_DIRS=/cds/sw/ds/ana/conda2/inst/envs/
    export DIR_PSDM=/cds/group/psdm
    export SIT_PSDM_DATA=/cds/data/psdm
    export SUBMODULEDIR=/cds/sw/ds/ana/conda2/rel/lcls2_submodules_02202026

    osrel=`uname -r`
    case $osrel in
        *el9*) conda activate daq_20250402_r9;;
        *)     conda activate daq_20250402;;
    esac

    # DAQ bundle from the active default environment
    export DAQ_CONDA_PREFIX=${CONDA_PREFIX}
    export DAQ_CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV}

    # AMI bundle: keep DAQ as active shell env, resolve AMI prefix by name
    export AMI_CONDA_DEFAULT_ENV=ps_20241122
    AMI_PREFIX_RESOLVED=$(conda info --envs 2>/dev/null | awk -v env_name="${AMI_CONDA_DEFAULT_ENV}" '$1 == env_name {print $NF; exit}')
    if [ -n "${AMI_PREFIX_RESOLVED}" ]; then
        export AMI_CONDA_PREFIX=${AMI_PREFIX_RESOLVED}
    else
        echo "Warning: conda env ${AMI_CONDA_DEFAULT_ENV} not found; using DAQ_CONDA_PREFIX for AMI_CONDA_PREFIX"
        export AMI_CONDA_PREFIX=${DAQ_CONDA_PREFIX}
    fi
    unset AMI_PREFIX_RESOLVED
elif [ -d "/sdf/group/lcls/" ]; then
    # for s3df
    source /sdf/group/lcls/ds/ana/sw/conda2-v4/inst/etc/profile.d/conda.sh
    export CONDA_ENVS_DIRS=/sdf/group/lcls/ds/ana/sw/conda2/inst/envs
    export DIR_PSDM=/sdf/group/lcls/ds/ana/
    export SIT_PSDM_DATA=/sdf/data/lcls/ds/
    conda activate xpp_drp_cpu_311_dev
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

<<<<<<< HEAD
export CUDA_ROOT=/usr/local/cuda
if [ -h "$CUDA_ROOT" ]; then
    export PATH=${CUDA_ROOT}/bin${PATH:+:${PATH}}
    #export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    export MANPATH=${CUDA_ROOT}/man${MANPATH:+:${MANPATH}}
fi

# In ASC lab the command for zsh does not work, but in XPP it does. 
# So we need to check which shell we are in to get the correct path to the script
if [ -n "$BASH_VERSION" ]; then
    SCRIPT_SOURCE="${BASH_SOURCE[0]}"
elif [ -n "$ZSH_VERSION" ]; then
    SCRIPT_SOURCE="${(%):-%x}"
fi

RELDIR="$(cd "$(dirname "$(readlink -f "$SCRIPT_SOURCE")")" && pwd)"
=======
RELDIR="$( cd "$( dirname $(readlink -f "${BASH_SOURCE[0]:-${(%):-%x}}") )" && pwd )"
>>>>>>> 5e69d1572 (WIP: updated build scripts to use meson)
export PATH=$RELDIR/install/bin:${PATH}
echo $RELDIR

pyver=$(python -c "import sys; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))")
export PYTHONPATH=$RELDIR/install/lib/python$pyver/site-packages
# for procmgr
export TESTRELDIR=$RELDIR/install

# DAQ/AMI TESTRELDIR bundles are currently used only in /cds/sw/ flow above
if [ -n "${DAQ_CONDA_PREFIX:-}" ]; then
    export DAQ_TESTRELDIR=${TESTRELDIR}
    export AMI_TESTRELDIR=${DAQ_TESTRELDIR}
fi

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

# needed by Ric to get correct libfabric man pages
export MANPATH=$CONDA_PREFIX/share/man${MANPATH:+:${MANPATH}}
