#!/bin/bash
#SBATCH --partition=anaq
#SBATCH --job-name=psbatch
#SBATCH --nodelist=drp-srcf-cmp036
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
t_start=`date +%s`
set -xe
srun -N1 -n1 procstat /cds/home/m/monarin/lcls2/psdaq/psdaq/slurm/p4.cnf.last -p 4
wait
t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start))