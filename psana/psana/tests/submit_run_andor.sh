#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=lcls:data
#SBATCH --job-name=run_andor
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --output=output-%j.txt
#SBATCH --error=output-%j.txt
##SBATCH --exclusive
#SBATCH -t 00:05:00

t_start=`date +%s`

#exp=rixc00221 
#runnum=49
exp=$1
runnum=$2
#srun python run_andor.py $exp $runnum ${socket}
mpirun -n 5 python run_andor.py $exp $runnum

t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start)) 
