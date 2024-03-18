#!/bin/bash
#SBATCH --partition=anaq
#SBATCH --job-name=test-slurm
#SBATCH --nodelist=drp-srcf-cmp035
#SBATCH --ntasks=2
##SBATCH --ntasks-per-node=50
#SBATCH --output=%j.log
#SBATCH --exclusive
#SBATCH --time=00:05:00
 

t_start=`date +%s`
set -xe
set -m

srun -n1 "sleep 5; echo hello1" & 
srun -n1 "sleep 10; echo hello2" & 
wait

t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start)) 
