source /cds/sw/ds/ana/conda2/manage/bin/psconda.sh
conda deactivate

conda activate ps-4.5.17
RELDIR="/cds/home/opr/rixopr/git/lcls2_100422"

# Before the move to SRCF
#conda activate ps-4.5.16
#RELDIR="/cds/home/opr/rixopr/git/lcls2_080522"

# in use until August 5, 2022
#conda activate ps-4.5.16
#RELDIR="/cds/home/opr/tmoopr/git/lcls2_072122"

# in use until July 21, 2022
#conda activate ps-4.5.16
#RELDIR="/cds/home/opr/rixopr/git/lcls2_062922"

# in use until June 29, 2022
#conda activate ps-4.5.11
#RELDIR="/cds/home/opr/rixopr/git/lcls2_051022"

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
