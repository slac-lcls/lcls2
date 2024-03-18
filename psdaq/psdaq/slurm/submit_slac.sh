#!/bin/bash

#SBATCH --job-name parallel   ## name that will show up in the queue
#SBATCH --output slurm-%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --ntasks=3  ## number of tasks (analyses) to run
##SBATCH --ntasks-per-node=2
#SBATCH --nodelist=drp-srcf-cmp036,drp-srcf-cmp035
##SBATCH --cpus-per-task=2  ## the number of threads allocated to each task
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH --partition=anaq  ## the partitions to run in (comma seperated)
#SBATCH --time=0-00:10:00  ## time for analysis (day-hour:min:sec)

# Execute job steps
srun --ntasks=1 --nodes=1 --exclusive --nodelist=drp-srcf-cmp036 bash -c "sleep 2; python test_mpi.py" &
srun --ntasks=1 --nodes=1 --exclusive --nodelist=drp-srcf-cmp036 bash -c "sleep 3; python test_mpi.py" &
srun --ntasks=1 --nodes=1 --exclusive --nodelist=drp-srcf-cmp035 bash -c "sleep 4; python test_mpi.py" &


wait

