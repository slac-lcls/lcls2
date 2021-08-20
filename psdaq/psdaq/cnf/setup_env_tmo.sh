source /cds/sw/ds/ana/conda2/manage/bin/psconda.sh
conda deactivate

conda activate ps-4.3.2
RELDIR="/cds/home/opr/tmoopr/git/lcls2_080421"

# official running until Aug 4, 2021
#conda activate ps-4.3.2
#RELDIR="/cds/home/opr/tmoopr/git/lcls2_060821"

# official running until June 5, 2021
#conda activate ps-4.3.2
#RELDIR="/cds/home/opr/tmoopr/git/lcls2_052621"

# official running until May 26, 2021
#conda activate ps-4.3.2
#RELDIR="/cds/home/opr/tmoopr/git/lcls2_051821"

export PATH=$RELDIR/install/bin:${PATH}
pyver=$(python -c "import sys; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))")
export PYTHONPATH=$RELDIR/install/lib/python$pyver/site-packages
# for procmgr
export TESTRELDIR=$RELDIR/install
