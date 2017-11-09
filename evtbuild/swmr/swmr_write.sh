#!/bin/bash
#source /reg/g/psdm/etc/psconda.sh
#source activate snek


python swmr_write.py


#`which mpirun` --oversubscribe -n 31 -H drp-tst-acc01,drp-tst-acc02 swmr_write.sh
#`which mpirun` --oversubscribe -n 32 -H drp-tst-acc03,drp-tst-acc04 swmr_read.sh
#`which mpirun` --oversubscribe -n 32 -H drp-tst-acc05,drp-tst-acc06 swmr_read_nersc.sh
