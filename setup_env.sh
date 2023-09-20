unset LD_LIBRARY_PATH
unset PYTHONPATH

if [[ ${HOSTNAME} == sdf* ]]
then
    # for s3df
    source /sdf/group/lcls/ds/ana/sw/conda2/inst/etc/profile.d/conda.sh
    export CONDA_ENVS_DIRS=/sdf/group/lcls/ds/ana/sw/conda2/inst/envs
    export DIR_PSDM=/sdf/group/lcls/ds/ana/
    export SIT_PSDM_DATA=/sdf/data/lcls/ds/
else
    # for psana
    source /cds/sw/ds/ana/conda2-v2/inst/etc/profile.d/conda.sh
    export CONDA_ENVS_DIRS=/cds/sw/ds/ana/conda2/inst/envs/
    export DIR_PSDM=/cds/group/psdm
    export SIT_PSDM_DATA=/cds/data/psdm
fi

source ~/.local/spack/activate_ps_2023-01-11_old_p4p.sh
export CC=$(which gcc)
export CXX=(which cxx)

AUTH_FILE=$DIR_PSDM"/sw/conda2/auth.sh"
if [ -f "$AUTH_FILE" ]; then
    source $AUTH_FILE
else
  echo "$AUTH_FILE file is missing"
fi

RELDIR="$( cd "$( dirname $(readlink -f "${BASH_SOURCE[0]}") )" && pwd )"
export PATH=$RELDIR/install/bin:${PATH}
pyver=$(python -c "import sys; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))")
export PYTHONPATH=$RELDIR/install/lib/python$pyver/site-packages:$PYTHONPATH
# for procmgr
export TESTRELDIR=$RELDIR/install
export PROCMGR_EXPORT=RDMAV_FORK_SAFE=1,RDMAV_HUGEPAGES_SAFE=1  # See fi_verbs man page regarding fork()
export PROCMGR_EXPORT=$PROCMGR_EXPORT,OPENBLAS_NUM_THREADS=1,PS_PARALLEL='none'

# cpo: seems that in more recent versions blas is creating many threads
export OPENBLAS_NUM_THREADS=1
# cpo: getting intermittent file-locking issue on ffb, so try this
export HDF5_USE_FILE_LOCKING=FALSE
# for libfabric. decreases performance a little, but allows forking
export RDMAV_FORK_SAFE=1
export RDMAV_HUGEPAGES_SAFE=1

# cpo: workaround a qt bug which may no longer be there (dec 5, 2022)
if [ ! -d /usr/share/X11/xkb ]; then
    export QT_XKB_CONFIG_ROOT=${CONDA_PREFIX}/lib
fi

# needed by Ric to get correct libfabric man pages
export MANPATH=$CONDA_PREFIX/share/man${MANPATH:+:${MANPATH}}
