unset LD_LIBRARY_PATH
unset PYTHONPATH
unset DAQ_CONDA_PREFIX
unset DAQ_CONDA_DEFAULT_ENV
unset DAQ_TESTRELDIR
unset AMI_CONDA_PREFIX
unset AMI_CONDA_DEFAULT_ENV
unset AMI_TESTRELDIR

if [ -d "/sdf/group/lcls/" ]
then
    # for s3df
    source /sdf/group/lcls/ds/ana/sw/conda2-v4/inst/etc/profile.d/conda.sh
    export CONDA_ENVS_DIRS=/sdf/group/lcls/ds/ana/sw/conda2/inst/envs
    export DIR_PSDM=/sdf/group/lcls/ds/ana/
    export SIT_PSDM_DATA=/sdf/data/lcls/ds/
else
    # for cds
    source /cds/sw/ds/ana/conda2-v4/inst/etc/profile.d/conda.sh
    export CONDA_ENVS_DIRS=/cds/sw/ds/ana/conda2/inst/envs/
    export DIR_PSDM=/cds/group/psdm
    export SIT_PSDM_DATA=/cds/data/psdm
fi

conda activate psana_20260405

# In ASC lab the command for zsh does not work, but in XPP it does.
# So we need to check which shell we are in to get the correct path to the script
if [ -n "$BASH_VERSION" ]; then
    SCRIPT_SOURCE="${BASH_SOURCE[0]}"
elif [ -n "$ZSH_VERSION" ]; then
    SCRIPT_SOURCE="${(%):-%x}"
fi

RELDIR="$(cd "$(dirname "$(readlink -f "$SCRIPT_SOURCE")")" && pwd)"
export PATH=$RELDIR/install/bin:${PATH}
pyver=$(python -V 2>&1 | grep -oP '\d+\.\d+' | head -1)
export PYTHONPATH=$RELDIR/install/lib/python$pyver/site-packages
export TESTRELDIR=$RELDIR/install

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
