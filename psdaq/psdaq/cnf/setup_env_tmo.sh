unset LD_LIBRARY_PATH
unset PYTHONPATH

# these three lines should be customized for the site
source /cds/sw/ds/ana/conda2-v2/inst/etc/profile.d/conda.sh
export CONDA_ENVS_DIRS=/cds/sw/ds/ana/conda2/inst/envs/
conda activate ps-4.5.26

RELDIR="$( cd "$( dirname $(readlink -f "${BASH_SOURCE[0]}") )" && pwd )"
export PATH=$RELDIR/install/bin:${PATH}
pyver=$(python -c "import sys; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))")
export PYTHONPATH=$RELDIR/install/lib/python$pyver/site-packages
# for procmgr
export TESTRELDIR=$RELDIR/install
export PROCMGR_EXPORT=RDMAV_FORK_SAFE=1,RDMAV_HUGEPAGES_SAFE=1  # See fi_verbs man page regarding fork()

# cpo: seems that in more recent versions blas is creating many threads
export OPENBLAS_NUM_THREADS=1
# cpo: getting intermittent file-locking issue on ffb, so try this
export HDF5_USE_FILE_LOCKING=FALSE
# for libfabric. decreases performance a little, but allows forking
export RDMAV_FORK_SAFE=1
export RDMAV_HUGEPAGES_SAFE=1
# Mikhail: root of psdm directories
export DIR_PSDM=/cds/group/psdm

# cpo: workaround a qt bug which may no longer be there (dec 5, 2022)
if [ ! -d /usr/share/X11/xkb ]; then
    export QT_XKB_CONFIG_ROOT=${CONDA_PREFIX}/lib
fi

# needed by Ric to get correct libfabric man pages
export MANPATH=$CONDA_PREFIX/share/man${MANPATH:+:${MANPATH}}
