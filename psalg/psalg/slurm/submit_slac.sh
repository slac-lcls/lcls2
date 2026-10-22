#!/bin/bash
#SBATCH --partition=drpq
#SBATCH --job-name=main
#SBATCH --ntasks=1
#SBATCH --x11=batch

srun --x11 xterm -hold -e python test_run.py
