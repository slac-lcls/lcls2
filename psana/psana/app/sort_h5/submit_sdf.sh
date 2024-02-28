#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=lcls:data
#SBATCH --job-name=test-psana2-ts-sort
#SBATCH --output=output-%j.txt
#SBATCH --error=output-%j.txt
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:10:00


t_start=`date +%s`

#timestamp_sort_h5
python -u main.py /sdf/data/lcls/drpsrcf/ffb/users/monarin/h5/mylargeh5.h5 /sdf/data/lcls/drpsrcf/ffb/users/monarin/h5/output/result.h5


t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start)) 
