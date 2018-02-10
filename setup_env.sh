source /reg/g/psdm/sw/conda2/manage/bin/psconda.sh
export PATH=`pwd`/install/bin:${PATH}
# temporary, until we install psdaq binaries
export PATH=`pwd`/build/bin:${PATH}
pyver=$(python -c "import sys; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))")
export PYTHONPATH=`pwd`/install/lib/python$pyver/site-packages
