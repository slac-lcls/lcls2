source /cds/sw/ds/ana/conda2/manage/bin/psconda.sh
conda deactivate

conda activate ps-4.5.10
RELDIR="/cds/home/opr/tmoopr/git/lcls2_021522"

# official running until Feb 15, 2022
#conda activate ps-4.5.10
#RELDIR="/cds/home/opr/tmoopr/git/lcls2_012522"

# official running until Jan 24, 2022
#conda activate ps-4.5.10
#RELDIR="/cds/home/opr/tmoopr/git/lcls2_012022"

# official running until Jan 21, 2022
#conda activate ps-4.5.10
#RELDIR="/cds/home/opr/tmoopr/git/lcls2_011422"

# official running until Jan 6, 2022
#conda activate ps-4.5.5
#RELDIR="/cds/home/opr/tmoopr/git/lcls2_120821"

# official running until Dec 8, 2021
#conda activate ps-4.5.5
#RELDIR="/cds/home/opr/tmoopr/git/lcls2_100621"

# official running until Oct 6, 2021
#conda activate ps-4.5.5
#RELDIR="/cds/home/opr/tmoopr/git/lcls2_090821"

# official running until Sep 8, 2021
#conda activate ps-4.3.2
#RELDIR="/cds/home/opr/tmoopr/git/lcls2_080421"

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
