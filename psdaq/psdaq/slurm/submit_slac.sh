#!/bin/bash
#SBATCH --partition=anaq
#SBATCH --job-name=test-slurm
#SBATCH --nodelist=drp-srcf-cmp[035-036]
#SBATCH --ntasks=20
##SBATCH --ntasks-per-node=50
#SBATCH --output=%j.log
#SBATCH --exclusive
#SBATCH --time=00:05:00
 

t_start=`date +%s`
set -xe


## Launch on "computes" node
SRUN1="srun -N1 -n1 --nodelist=drp-srcf-cmp035"

$SRUN1 drp -P tst -C drp-srcf-cmp035 -M /cds/group/psdm/psdatmgr/etc/config/prom/tst -d /dev/datadev_1 -o /cds/data/drpsrcf -k batching=yes,directIO=yes -l 0x1 -D ts -u timing_0 -p 4 > log_drp

$SRUN1 teb -P tst -C drp-srcf-cmp035 -M /cds/group/psdm/psdatmgr/etc/config/prom/tst -k script_path=/cds/home/c/claus/lclsii/daq/runs/eb/data/srcf -u teb0 -p 4 > log_teb &

$SRUN1 control -P tst -B DAQ:NEH-x0-CBEAM --user tstopr --url https://pswww.slac.stanford.edu/ws-auth/lgbk/ -d https://pswww.slac.stanford.edu/ws-auth/configdb/ws/configDB -t trigger -S 1 -T 20000 -V /cds/group/pcds/dist/pds/tst/misc/elog_tst.txt -u control -p 4 > log_control &

$SRUN1 procServ --noautorestart --name 2024/03/05_11:26:49_drp-srcf-cmp035:timing_0.log --pidfile /dev/null -L /cds/home/m/monarin/2024/03/05_11:26:49_drp-srcf-cmp035:timing_0.log --allow 29406/bin/envLD_LIBRARY_PATH=/cds/sw/ds/ana/conda2/inst/envs/ps-4.6.3/epics/lib/linux-x86_64:/cds/sw/ds/ana/conda2/inst/envs/ps-4.6.3/pcas/lib/linux-x86_64/usr/bin/taskset-c4-63drp-Ptst-Cdrp-srcf-cmp035 -M /cds/group/psdm/psdatmgr/etc/config/prom/tst-d/dev/datadev_1 -o /cds/data/drpsrcf -k batching=yes,directIO=yes -l 0x1 -D ts -u timing_0 -p 4 > log_procserv_01 &

$SRUN1 procServ --noautorestart --name 2024/03/05_11:26:49_drp-srcf-cmp035:teb0.log --pidfile /dev/null -L /cds/home/m/monarin/2024/03/05_11:26:49_drp-srcf-cmp035:teb0.log --allow 29405/usr/bin/taskset-c4-63teb-Ptst-Cdrp-srcf-cmp035 -M /cds/group/psdm/psdatmgr/etc/config/prom/tst -k script_path=/cds/home/c/claus/lclsii/daq/runs/eb/data/srcf -u teb0 -p 4 > log_procserv_02 &

$SRUN1 procServ --noautorestart --name 2024/03/05_11:26:49_drp-srcf-cmp035:control.log --pidfile /dev/null -L /cds/home/m/monarin/2024/03/05_11:26:49_drp-srcf-cmp035:control.log --allow 29403/bin/envLD_LIBRARY_PATH=/cds/sw/ds/ana/conda2/inst/envs/ps-4.6.3/epics/lib/linux-x86_64:/cds/sw/ds/ana/conda2/inst/envs/ps-4.6.3/pcas/lib/linux-x86_64/cds/home/m/monarin/lcls2/install/bin/control-Ptst-BDAQ:NEH-x0-CBEAM --user tstopr --url https://pswww.slac.stanford.edu/ws-auth/lgbk/ -d https://pswww.slac.stanford.edu/ws-auth/configdb/ws/configDB -t trigger -S 1 -T 20000 -V /cds/group/pcds/dist/pds/tst/misc/elog_tst.txt -u control -p 4 > log_procserv_03 &



## Launch on "local" node
#srun -N1 -n1 --nodelist=drp-srcf-cmp035 control_gui -H drp-srcf-cmp035 --uris https://pswww.slac.stanford.edu/ws-auth/configdb/ws --expert --user tstopr --loglevel WARNING -p 4


t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start)) 
