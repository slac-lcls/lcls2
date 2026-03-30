#!/bin/bash

## sbatch ./lcls2/psana/psana/app/jungfrau_dark_proc_sbatch.sh # execute this script with default command
## sbatch --wait --ntasks-per-node 19 jungfrau_dark_proc_sbatch.sh "jungfrau_dark_proc -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 --nrecs 1000 --nrecs1 0"
## sbatch --nodes=1 --ntasks-per-node=19 jungfrau_dark_proc_sbatch.sh "jungfrau_dark_proc -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 --nrecs 1000 --nrecs1 0"
## squeue -u dubrovin # check batch jobs in queue
## scancel [JobID]
## scontrol show jobid <job_id>

#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=19
#SBATCH --exclusive
#SBATCH --output=log_jungfrau_dark_proc_sbatch_%j.log
#SBATCH --account=lcls:prjdat21

echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NPROCS: $SLURM_NPROCS"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_CPUS_ON_NODE: $SLURM_CPUS_ON_NODE"
echo "SLURM_JOB_CPUS_PER_NODE: $SLURM_JOB_CPUS_PER_NODE"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# -u flushes print statements which can otherwise be hidden if mpi hangs and ?-m mpi4py.run? can prevent some hanging jobs
#mpirun --mca osc ^ucx python -u -m mpi4py.run $cmd

# optional $1 positional argument for script - executed in sbatch command:
cmd_def="jungfrau_dark_proc -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 --nrecs 1000 --nrecs1 0"
command="${1:-$cmd_def}"
echo "command: $command"

mpirun --mca osc ^ucx $command


