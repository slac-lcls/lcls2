#!/bin/bash

#SBATCH --partition=anaq  
#SBATCH --job-name parallel   
#SBATCH --output slurm-%j.out   
#SBATCH --ntasks=1  
#SBATCH --nodelist=drp-srcf-cmp036

# Execute job steps
srun -n1 --export=ALL,BLAH=blah python $HOME/psana-nersc/psana2/test_mpi.py



