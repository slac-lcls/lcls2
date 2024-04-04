#!/bin/bash

#SBATCH --partition=drpq
#SBATCH --job-name parallel   
#SBATCH --output slurm-%j.out   
#SBATCH -N1 -n3 --nodelist=drp-srcf-cmp035 
#SBATCH hetjob
#SBATCH -N1 -n1 --nodelist=drp-srcf-cmp036


# Execute job steps
export BLAH="blah"
srun -n 1 --het-group=0 --exclusive bash -c "sleep 5; echo $BLAH 1" &
srun -n 1 --het-group=0 --exclusive bash -c "sleep 10; echo $BLAH 2" &
srun -n 1 --het-group=0 --exclusive bash -c "sleep 15; echo $BLAH 3" &

srun -n 1 --het-group=1 --exclusive bash -c "sleep 15; echo $BLAH 4" &
wait


