source /cds/sw/ds/ana/conda2/manage/bin/psconda.sh

#conda deactivate
conda activate ps-4.5.11
RELDIR="/cds/home/opr/rixopr/git/lcls2_051022"

# in use until May 23, 2022
#conda activate ps-4.5.10
#RELDIR="/cds/home/opr/rixopr/git/lcls2_021522"

# in use until Feb 15, 2022
#conda activate ps-4.5.5
#RELDIR="/cds/home/opr/rixopr/git/lcls2_012022"

export PATH=$RELDIR/install/bin:${PATH}
pyver=$(python -c "import sys; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))")
export PYTHONPATH=$RELDIR/install/lib/python$pyver/site-packages
# for procmgr
export TESTRELDIR=$RELDIR/install
