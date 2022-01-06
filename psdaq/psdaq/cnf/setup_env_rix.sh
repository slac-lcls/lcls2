source /cds/sw/ds/ana/conda2/manage/bin/psconda.sh

conda activate ps-4.5.5
RELDIR="/cds/home/opr/rixopr/git/lcls2_120821"

# in use until Dec. 8, 2021
#conda activate ps-4.5.5
#RELDIR="/cds/home/opr/rixopr/git/lcls2_111921"

# in use until Nov 21, 2021
#conda activate ps-4.5.5
#RELDIR="/cds/home/opr/rixopr/git/lcls2_102821"

# in use until Nov 2, 2021
#conda activate ps-4.5.5
#RELDIR="/cds/home/opr/rixopr/git/lcls2_100621"

# in use until Oct 6, 2021
#conda activate ps-4.5.5
#RELDIR="/cds/home/opr/rixopr/git/lcls2_100421"

# in use until Oct 4, 2021
#conda activate ps-4.5.5
#RELDIR="/cds/home/opr/rixopr/git/lcls2_092121"

# in use until Sept 22, 2021
# conda activate ps-4.5.5
# RELDIR="/cds/home/opr/rixopr/git/lcls2_090921"

# in use until Sept. 15, 20201
#conda activate ps-4.3.2
#RELDIR="/cds/home/opr/rixopr/git/lcls2_080421"

#conda activate ps-4.5.2
#RELDIR="/cds/home/opr/rixopr/git/lcls2_070221_newrogue"

# in use until July 3, 2021
#conda activate ps-4.3.2
#RELDIR="/cds/home/opr/rixopr/git/lcls2_062121"
#RELDIR="/cds/home/opr/rixopr/git/lcls2_062121_pv"

# in use until June 21, 2021
#conda activate ps-4.3.2
#RELDIR="/cds/home/opr/rixopr/git/lcls2_061521"

# in use until June 15, 2021
#conda activate ps-4.3.2
#RELDIR="/cds/home/opr/rixopr/git/lcls2_060821"

# in use until June 8, 2021
#conda activate ps-4.3.2
#RELDIR="/cds/home/opr/rixopr/git/lcls2_052821"

export PATH=$RELDIR/install/bin:${PATH}
pyver=$(python -c "import sys; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))")
export PYTHONPATH=$RELDIR/install/lib/python$pyver/site-packages
# for procmgr
export TESTRELDIR=$RELDIR/install
