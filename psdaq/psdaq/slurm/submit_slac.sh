#!/bin/bash
#SBATCH --partition=drpq
#SBATCH --job-name=main
#SBATCH --output=/cds/home/m/monarin/2024/04/18_14:05:46_drp-srcf-cmp036:slurm.log
#SBATCH --nodelist=drp-srcf-cmp035 --ntasks=3
#SBATCH hetjob
#SBATCH --nodelist=drp-srcf-cmp036 --ntasks=1
srun -n1 --exclusive --job-name=timing_0 --het-group=0 --output=/cds/home/m/monarin/2024/04/18_14:05:46_drp-srcf-cmp035:timing_0.log --open-mode=append --export=ALL,LD_LIBRARY_PATH=/cds/sw/ds/ana/conda2/inst/envs/ps-4.6.3/epics/lib/linux-x86_64:/cds/sw/ds/ana/conda2/inst/envs/ps-4.6.3/pcas/lib/linux-x86_64  bash -c ' drp -P tst -M /cds/group/psdm/psdatmgr/etc/config/prom/tst -d /dev/datadev_1 -o /cds/data/drpsrcf -k batching=yes,directIO=yes -l 0x1 -D ts -p '4' -u timing_0 -C drp-srcf-cmp035'& 
srun -n1 --exclusive --job-name=control --het-group=0 --output=/cds/home/m/monarin/2024/04/18_14:05:46_drp-srcf-cmp035:control.log --open-mode=append --export=ALL,LD_LIBRARY_PATH=/cds/sw/ds/ana/conda2/inst/envs/ps-4.6.3/epics/lib/linux-x86_64:/cds/sw/ds/ana/conda2/inst/envs/ps-4.6.3/pcas/lib/linux-x86_64  bash -c 'control -P tst -B DAQ:NEH -x 0 -C BEAM  --user tstopr   --url https://pswww.slac.stanford.edu/ws-auth/lgbk/  -d https://pswww.slac.stanford.edu/ws-auth/configdb/ws/configDB -t trigger -S 1 -T 20000 -V /cds/group/pcds/dist/pds/tst/misc/elog_tst.txt -p '4' -u control'& 
srun -n1 --exclusive --job-name=teb0 --het-group=0 --output=/cds/home/m/monarin/2024/04/18_14:05:46_drp-srcf-cmp035:teb0.log --open-mode=append  bash -c ' teb -P tst -C drp-srcf-cmp035 -M /cds/group/psdm/psdatmgr/etc/config/prom/tst -k script_path=/cds/home/c/claus/lclsii/daq/runs/eb/data/srcf -p '4' -u teb0'& 
srun -n1 --exclusive --job-name=control_gui --het-group=1 --output=/cds/home/m/monarin/2024/04/18_14:05:46_drp-srcf-cmp036:control_gui.log --open-mode=append  bash -c 'control_gui --uris https://pswww.slac.stanford.edu/ws-auth/configdb/ws --expert  --user tstopr  --loglevel WARNING -p '4' -u control_gui -H drp-srcf-cmp035'& 
wait
