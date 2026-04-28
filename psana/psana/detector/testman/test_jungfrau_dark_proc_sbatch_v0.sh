#!/bin/bash

## sbatch ./lcls2/psana/psana/app/jungfrau_dark_proc_sbatch.sh # execute this script
## squeue -u dubrovin # check batch jobs in queue
## scancel [JobID]
## killall mpirun # does not work

#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=35
#SBATCH --exclusive
#SBATCH --output=jungfrau_dark_calib_%j.log
#SBATCH --account=lcls:prjdat21

#jdp="/sdf/home/d/dubrovin/LCLS/con-lcls2/lcls2/psana/psana/app/jungfrau_dark_proc.py"
# -u flushes print statements which can otherwise be hidden if mpi hangs and ?-m mpi4py.run? can prevent some hanging jobs
#mpirun --mca osc ^ucx python -u -m mpi4py.run $cmd

jdp="jungfrau_dark_proc"
cmd="$jdp -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 --nrecs 100 --nrecs1 0"
mpirun --mca osc ^ucx $cmd


