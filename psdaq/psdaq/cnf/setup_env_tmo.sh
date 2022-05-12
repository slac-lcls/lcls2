source /cds/sw/ds/ana/conda2/manage/bin/psconda.sh
conda deactivate

conda activate ps-4.5.10
RELDIR="/cds/home/opr/tmoopr/git/lcls2_041322"

# official running until Apr 13, 2022
#conda activate ps-4.5.10
#RELDIR="/cds/home/opr/tmoopr/git/lcls2_022222"

# official running until Feb 22, 2022
#conda activate ps-4.5.10
#RELDIR="/cds/home/opr/tmoopr/git/lcls2_021522"

# official running until Feb 15, 2022
#conda activate ps-4.5.10
#RELDIR="/cds/home/opr/tmoopr/git/lcls2_012522"

export PATH=$RELDIR/install/bin:${PATH}
pyver=$(python -c "import sys; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))")
export PYTHONPATH=$RELDIR/install/lib/python$pyver/site-packages
# for procmgr
export TESTRELDIR=$RELDIR/install
