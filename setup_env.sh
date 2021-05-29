unset LD_LIBRARY_PATH
unset PYTHONPATH
source /cds/sw/ds/ana/conda2/inst/etc/profile.d/conda.sh
# kludge: somehow conda activate is executing rogue stuff before
# LD_LIBRARY_PATH is set to pick up the ugly location of epics
export LD_LIBRARY_PATH=/cds/sw/ds/ana/conda2/inst/envs/ps-4.5.1/epics/lib/linux-x86_64:/cds/sw/ds/ana/conda2/inst/envs/ps-4.5.1/pcas/lib/linux-x86_64
conda activate  ps-4.5.1
RELDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PATH=$RELDIR/install/bin:${PATH}
pyver=$(python -c "import sys; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))")
export PYTHONPATH=$RELDIR/install/lib/python$pyver/site-packages
# for procmgr
export TESTRELDIR=$RELDIR/install

# needed by Ric to get correct libfabric man pages
export MANPATH=$CONDA_PREFIX/share/man${MANPATH:+:${MANPATH}}
