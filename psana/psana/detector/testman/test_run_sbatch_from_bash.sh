#!/usr/bin/bash

# Submits command below from bash:
# sbatch --partition=milano --account=lcls:prjdat21 --export=ALL --output=log_test_run_sbatch_from_bash_%j.log --nodes=1 --ntasks-per-node=19 lcls2/psana/psana/app/jungfrau_dark_proc_sbatch.sh jungfrau_dark_proc -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 --nrecs 100 --nrecs1 0
# 0) cd <path>/lcls2/..
# 1) jungfrau_dark_proc -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 --nrecs 50 --nrecs1 50
# 2) ./psana/psana/detector/testman/test_run_sbatch_from_bash.sh

#logfile="log_test_run_sbatch_from_bash_%j.log"
logfile="$(date +%Y-%m-%dT%H%M%S)_log_test_run_sbatch_from_bash_$(whoami).log"
cmd="jungfrau_dark_proc -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 --nrecs 100 --nrecs1 0"
file="lcls2/psana/psana/app/jungfrau_dark_proc_sbatch.sh"

if [ -f "$file" ]; then
    echo "file $file is available"
else
    echo "file $file DOES NOT EXIST or is not a regular file. CHANGE PATH TO THIS FILE"
fi

### cmd_sbatch="sbatch --export=ALL --ntasks-per-node 19 $file \"$cmd\"" # DOES NOT WORK!!!
### echo "$cmd_sbatch"
### $cmd_sbatch

cmd_sbatch=(sbatch --partition=milano --account=lcls:prjdat21 --export=ALL --output=$logfile --nodes=1 --ntasks-per-node=19 "$file" "$cmd")

echo; echo "run command from this bash: ${cmd_sbatch[@]}"

"${cmd_sbatch[@]}"

echo; echo "SEE LOG FILE: $logfile"; echo
# EOF
