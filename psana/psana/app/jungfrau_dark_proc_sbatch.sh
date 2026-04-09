#!/bin/bash

## sbatch ./lcls2/psana/psana/app/jungfrau_dark_proc_sbatch.sh # execute this script with default command
## sbatch --wait --ntasks-per-node 19 jungfrau_dark_proc_sbatch.sh "jungfrau_dark_proc -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 --nrecs 1000 --nrecs1 0"
## sbatch --nodes=1 --ntasks-per-node=19 jungfrau_dark_proc_sbatch.sh "jungfrau_dark_proc -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 --nrecs 1000 --nrecs1 0"
## squeue -u dubrovin # check batch jobs in queue
## scancel [JobID]
## scontrol show jobid <job_id>

logfile="$(date +%Y-%m-%dT%H%M%S)_jungfrau_dark_proc_sbatch_$(whoami)_%j.log"

#SBATCH --partition=milano
#SBATCH --account=lcls:prjdat21
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=19
#SBATCH --output=$logfile

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

# optional $# positional arguments or their default value:
cmd11="${1:-NONE}"
cmd12="${2:-NONE}"
cmd13="${3:-NONE}"
cmd20="${4:-NONE}"
dirrepo="${5:-./work1}"

t0_sec=$SECONDS

if [[ $cmd11 != "NONE" ]]; then $cmd11 >/dev/null 2>&1 & pid11=$!; echo "started PID $pid11 for command: $cmd11"; fi
if [[ $cmd12 != "NONE" ]]; then $cmd12 >/dev/null 2>&1 & pid12=$!; echo "started PID $pid12 for command: $cmd12"; fi
if [[ $cmd13 != "NONE" ]]; then echo "run command: $cmd13"; echo; $cmd13; fi

if true; then
    echo
    for p in {$pid11,$pid12}; do
      if ps -p $p > /dev/null; then state="RUNNING"; else state="COMPLETED"; fi
      echo "process $p is $state"
    done

    if [[ $cmd11 != "NONE" ]]; then
	echo "skip cmd11,2,3: $cmd11, $cmd12, $cmd12";
    else
	echo "STAGE 1 IS COMPLETED time for three steps: $((SECONDS - t0_sec)) sec"
    fi
    echo
fi

if true; then
  echo
  echo "CHECK THAT ALL 6 ARRAYS WITH GATE LIMITS ARE AVAILABLE"
  cmddir="ls -ltr $dirrepo/jungfrau/block_results/"
  echo $cmddir
  $cmddir
  echo
fi

if [[ $cmd20 != "NONE" ]]; then
    t0_sec_stage2=$SECONDS
    echo "start: mpirun --mca osc ^ucx $cmd20"
    mpirun --mca osc ^ucx $cmd20
    echo "sbatch STAGE 2 IS COMPLETED time: $((SECONDS - t0_sec_stage2)) sec"
else
    echo "skip cmd20: $cmd20"
fi

#if [[ $cmd30 != "NONE" ]]; then
#    t0_sec_stage3=$SECONDS
#    echo "run command cmd30: $cmd30";
#    $cmd30;
#    echo "sbatch STAGE 3 IS COMPLETED time: $((SECONDS - t0_sec_stage3)) sec"
#else
#    echo "skip cmd30: $cmd30";
#fi

echo "sbatch job TOTAL TIME: $((SECONDS - t0_sec)) sec"

# EOF
