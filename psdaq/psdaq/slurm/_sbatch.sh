#!/bin/bash
#SBATCH --partition=anaq
#SBATCH --job-name=psbatch
#SBATCH --nodelist=drp-srcf-cmp035
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
t_start=`date +%s`
set -xe
srun -N1 -n1 drp -P tst -C drp-srcf-cmp035 -M /cds/group/psdm/psdatmgr/etc/config/prom/tst -d /dev/datadev_1 -o /cds/data/drpsrcf -k batching=yes,directIO=yes -l 0x1 -D ts -u timing_0 -p 4
echo $SLURM_JOB_ID
t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start))