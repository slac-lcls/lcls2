#!/bin/bash
############################################################
# First node must be exclusive to smd0
# * For openmpi, slots=1 must be assigned to the first node.
############################################################

# Get list of hosts by expand shorthand node list into a
# line-by-line node list
host_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
hosts=('$host_list')

# Write out to host file by putting rank 0 on the first node
host_file="slurm_host_${SLURM_JOB_ID}"
for i in "${!hosts[@]}"; do
    if [[ "$i" == "0" ]]; then
        echo "${hosts[$i]}" slots=1 > $host_file
    else
        echo "${hosts[$i]}" >> $host_file
    fi
done

# Export hostfile for mpirun
export PS_HOST_FILE=$host_file

# Calculate no. of ranks available in the job.
export PS_N_RANKS=$(( SLURM_CPUS_ON_NODE * ( SLURM_JOB_NUM_NODES - 1 ) + 1 ))
