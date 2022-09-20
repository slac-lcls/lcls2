source /cds/sw/ds/ana/conda2/manage/bin/psconda.sh

conda activate ps-4.5.10
RELDIR="/cds/home/opr/uedopr/git/lcls2_220502"

# in production until May 2, 2022
#RELDIR="/cds/home/opr/uedopr/git/lcls2_220420"

# in production until Apr 20, 2022
#RELDIR="/cds/home/opr/uedopr/git/lcls2_220418"

# in production until Apr 18, 2022
#RELDIR="/cds/home/opr/tmoopr/git/lcls2_041522"

# in production until Apr 15, 2022
#conda activate ps-4.5.10
#RELDIR="/cds/home/opr/uedopr/git/lcls2_220408"

# in production until Apr 08, 2022
#RELDIR="/cds/home/opr/tmoopr/git/lcls2_040522"

# in production until Apr 05, 2022
#conda activate ps-4.5.10
#RELDIR="/cds/home/opr/tmoopr/git/lcls2_022522"

# in production until Feb 11, 2022
#conda activate ps-4.3.2
#RELDIR="/cds/home/opr/uedopr/git/lcls2_210605"

# in production until June 5, 2021
#RELDIR="/cds/home/opr/uedopr/git/lcls2_210514"

#RELDIR="/cds/home/opr/uedopr/git/lcls2_210323"
export PATH=$RELDIR/install/bin:${PATH}
pyver=$(python -c "import sys; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))")
export PYTHONPATH=$RELDIR/install/lib/python$pyver/site-packages
# for procmgr
export TESTRELDIR=$RELDIR/install

# Add some UED specific EPICS environment.
# This is needed for running control scanning scripts from drp-ued-cmp002.
export EPICS_CA_SERVER_PORT=5058
export EPICS_CA_ADDR_LIST="172.27.99.255 172.21.36.255:5064"
