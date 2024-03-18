#!/bin/bash
#SBATCH --partition=anaq
#SBATCH --job-name=psbatch
#SBATCH --nodelist=drp-srcf-cmp035,drp-srcf-cmp036
#SBATCH --ntasks=6
#SBATCH --time=00:15:00
t_start=`date +%s`
set -xe
srun -N1 -n1 --exclusive --nodelist=drp-srcf-cmp035 control -P tst -B DAQ:NEH -x 0 -C BEAM  --user tstopr   --url https://pswww.slac.stanford.edu/ws-auth/lgbk/  -d https://pswww.slac.stanford.edu/ws-auth/configdb/ws/configDB -t trigger -S 1 -T 20000 -V /cds/group/pcds/dist/pds/tst/misc/elog_tst.txt -u control -p 4& 
srun -N1 -n1 --exclusive --nodelist=drp-srcf-cmp035 taskset -c 4-63 teb -P tst -C drp-srcf-cmp035 -M /cds/group/psdm/psdatmgr/etc/config/prom/tst -k script_path=/cds/home/c/claus/lclsii/daq/runs/eb/data/srcf -u teb0 -p 4& 
srun -N1 -n1 --exclusive --nodelist=drp-srcf-cmp035 taskset -c 4-63 drp -P tst -C drp-srcf-cmp035 -M /cds/group/psdm/psdatmgr/etc/config/prom/tst -d /dev/datadev_1 -o /cds/data/drpsrcf -k batching=yes,directIO=yes -l 0x1 -D ts -u timing_0 -p 4& 
srun -N1 -n1 --exclusive --nodelist=drp-srcf-cmp036 control_gui -H drp-srcf-cmp035 --uris https://pswww.slac.stanford.edu/ws-auth/configdb/ws --expert  --user tstopr  --loglevel WARNING -p 4& 

wait
t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start))