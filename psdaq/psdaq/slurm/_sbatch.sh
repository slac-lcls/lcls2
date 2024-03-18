#!/bin/bash
#SBATCH --partition=anaq
#SBATCH --job-name=control_gui
#SBATCH --nodelist=drp-srcf-cmp036
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
t_start=`date +%s`
set -xe
srun -N1 -n1 control_gui -H drp-srcf-cmp035 --uris https://pswww.slac.stanford.edu/ws-auth/configdb/ws --expert  --user tstopr  --loglevel WARNING -p 4
wait
t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start))