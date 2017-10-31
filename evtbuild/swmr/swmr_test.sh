#!/bin/bash

`which mpirun` --oversubscribe -n 1 -H drp-tst-acc01 run_swmr_write.sh

sleep(5)
`which mpirun` --oversubscribe -n 1 -H drp-tst-acc02 run_swmr_read.sh


sleep(240)


