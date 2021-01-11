source /cds/sw/ds/ana/conda2/manage/bin/psconda.sh
conda activate ps-4.1.3
RELDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PATH=$RELDIR/install/bin:${PATH}
pyver=$(python -c "import sys; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))")
export PYTHONPATH=$RELDIR/install/lib/python$pyver/site-packages
# for procmgr
export TESTRELDIR=$RELDIR/install

# needed by Ric to get correct libfabric man pages
export MANPATH=$CONDA_PREFIX/share/man${MANPATH:+:${MANPATH}}
