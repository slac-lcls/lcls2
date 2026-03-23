#!/usr/bin/bash
####!/usr/bin/env zsh

# LOCAL PARAMETERS:
# --submit ### for debugging, show what script is doing, but DO NOT EXECUTE COMMANDS
# --stages ### ALL-for all, 1-for stage 1 ONLY, 2-...
# --nranks ### number of mpirun runks, if 1-no mpi

# lcls2/psana/psana/app/jungfrau_dark_proc_wrapper.sh -S 1 ### run stage 1 ONLY !
# lcls2/psana/psana/app/jungfrau_dark_proc_wrapper.sh -S 2 ### run stage 2 ONLY !
# jungfrau_dark_proc -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 --stages 7 --wrapper [--submit]

show_argv() {
    local i=0
    for arg in "$@"; do
        i=$((i + 1))
        printf '  %02d: %s\n' "$i" "$arg"
    done
}

# Initialize variables with default values
M14="0x3fff"
M14minus="0x3ffe"

dskwargs="exp=mfx100848724,run=49" # "exp=mfxdaq23,run=7,dir=/sdf/data/lcls/drpsrcf/ffb/MFX/mfxdaq23/xtc"
detname="jungfrau"
nrecs="100"
nrecs1="50"
#dirrepo="/sdf/group/lcls/ds/ana/detector/calib2/constants"
dirrepo="./work1"
logmode="INFO"
errskip="True"
stepnum="None"
stepmax="3"
evskip="0"
events="3000"  # goes to DataSource(..., max_events=events, ...)
evstep="1000"
dirmode="0o2775"
filemode="0o664"
group="ps-users"
int_lo="1"
int_hi="$M14minus"
intnlo="6.0"
intnhi="6.0"
rms_lo="0.001"
rms_hi="$M14minus"
rmsnlo="6.0"
rmsnhi="6.0"
fraclm="0.1"
fraclo="0.05"
frachi="0.95"
version="V2026-03-23"
datbits="$M14"
deploy=false
save=false
plotim="0"
segind="None"
nranks="19"   #"19" 1-no mpi
nnodes="1"
stages=7
submit=false # execute/skip commands for debudding of this script

# Loop through all arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -k|--dskwargs)
            dskwargs="$2"
            shift # Skip the value argument
            ;;
        -d|--detname)
            detname="$2"
            shift
            ;;
        -n|--nrecs)
            nrecs="$2"
            shift
            ;;
        --nrecs1)
            nrecs1="$2"
            shift
            ;;
        -o|--dirrepo)
            dirrepo="$2"
            shift
            ;;
        -L|--logmode)
            logmode="$2"
            shift
            ;;
        -E|--errskip)
            errskip="$2"
            shift
            ;;
        --stepnum)
            stepnum="$2"
            shift
            ;;
        --stepmax)
            stepmax="$2"
            shift
            ;;
        --evskip)
            evskip="$2"
            shift
            ;;
        --events)
            events="$2"
            shift
            ;;
        -e|--evstep)
            evstep="$2"
            shift
            ;;
        --dirmode)
            dirmode="$2"
            shift
            ;;
        --filemode)
            filemode="$2"
            shift
            ;;
        --int_lo)
            int_lo="$2"
            shift
            ;;
        --int_hi)
            int_hi="$2"
            shift
            ;;
        --intnlo)
            intnlo="$2"
            shift
            ;;
        --intnhi)
            intnhi="$2"
            shift
            ;;
        --rms_lo)
            rms_lo="$2"
            shift
            ;;
        --rms_hi)
            rms_hi="$2"
            shift
            ;;
        --rmsnlo)
            rmsnlo="$2"
            shift
            ;;
        --rmsnhi)
            rmsnhi="$2"
            shift
            ;;
        --fraclm)
            fraclm="$2"
            shift
            ;;
        --fraclo)
            fraclo="$2"
            shift
            ;;
        --frachi)
            frachi="$2"
            shift
            ;;
        -v|--version)
            version="$2"
            shift
            ;;
        --datbits)
            datbits="$2"
            shift
            ;;
        -D|--deploy)
            deploy=true
            #shift
            ;;
        -S|--save)
            save=true
            #shift
            ;;
        -p|--plotim)
            plotim="$2"
            shift
            ;;
        -I|--segind)
            segind="$2"
            shift
            ;;
        -N|--nranks)
            nranks="$2"
            shift
            ;;
        --nnodes)
            nnodes="$2"
            shift
            ;;
        -S|--stages)
            stages="$2"
            shift
            ;;
        --submit)
            submit=true
            #shift
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift # Skip the option argument
done

istages=$(($stages))
((stage1 = istages & 1)); [[ stage1 -gt 0 ]] && stage1=true || stage1=false
((stage2 = istages & 2)); [[ stage2 -gt 0 ]] && stage2=true || stage2=false
((stage3 = istages & 4)); [[ stage3 -gt 0 ]] && stage3=true || stage3=false

echo
echo "in $0"
echo "--stages $stages: do stages 1/2/3: $stage1/$stage2/$stage3"

script_dir=$(dirname "$(realpath "$0")")
echo "script_dir: $script_dir"


#c11="jungfrau_dark_proc -k $dskwargs -d $detname -n $nrecs --nrecs1 $nrecs1 -o $dirrepo -L $logmode -E $errskip --stepnum $stepnum --stepmax $stepmax --evskip $evskip"
#c12="--dirmode $dirmode --filemode $filemode --events $events -e $evstep"
#c13="--int_lo $int_lo --int_hi $int_hi --intnlo $intnlo --intnhi $intnhi --rms_lo $rms_lo --rms_hi $rms_hi --rmsnlo $rmsnlo --rmsnhi $rmsnhi"
#c14="--fraclm $fraclm --fraclo $fraclo --frachi $frachi -v $version --datbits $datbits -D $deploy -p $plotim -I $segind"
#c1="$c11 $c12 $c13 $c14"

c01="--datbits $datbits --int_lo $int_lo --int_hi $int_hi --fraclo $fraclo --frachi $frachi"
cmnpars="-k $dskwargs -d $detname -o $dirrepo -L $logmode"
cmd00="jungfrau_dark_proc $cmnpars" ### "jungfrau_dark_proc -k exp=mfx100848724,run=49 -d jungfrau -o ./work1"
cmd10="$cmd00 --nrecs $nrecs1 --nrecs1 $nrecs1 $c01"
cmd11="$cmd10 --stepnum 0"
cmd12="$cmd10 --stepnum 1"
cmd13="$cmd10 --stepnum 2"


#nevts=$(($nrecs * 3))
#echo "evaluate total number of events as $nevts"
#exit 1

#cmdmpi="mpirun --mca osc ^ucx -n $nranks"
#[[ "$nranks" != "1" ]] && cmdmpi="mpirun --mca osc ^ucx -n $nranks " || cmdmpi=""


#c02_proc="--events $events --evskip $evskip --stepnum $stepnum --stepmax $stepmax"
c02_proc="--evskip $evskip" # --stepnum $stepnum --stepmax $stepmax"
c02_status="--int_hi $int_hi --int_lo $int_lo --intnhi $intnhi --intnlo $intnlo --rms_hi $rms_hi --rms_lo $rms_lo --rmsnhi $rmsnhi --rmsnlo $rmsnlo --fraclm $fraclm"
cmd20="$cmd00 --nrecs $nrecs --nrecs1 0 $c02_proc $c02_status"
if $deploy; then cmd20="$cmd20 --deploy"; fi
if $save;   then cmd20="$cmd20 --save"; fi

#cmd30="$script_dir/jungfrau_deploy_constants.py $cmnpars -F" # -S -D
cmd30="jungfrau_deploy_constants $cmnpars -F" # -S -D

if [[ "$logmode" == "DEBUG" ]]; then
  echo "COMMAND LINES"
  echo "common : $cmd00"
  echo "stage 1: $cmd10"
  echo "stage 2: $cmd20"
  echo "stage 3: $cmd30"
fi

t0_sec=$SECONDS

#time {


if $stage1; then
  echo
  echo "=== STAGE 1 - evaluate intensity gates for $fraclo and $frachi part of statistics on $nrecs1 events"

  if $submit; then nohup $cmd11 >/dev/null 2>&1 & fi
  pid11=$!
  echo "started PID $pid11 for command: $cmd11"

  if $submit; then nohup $cmd12 >/dev/null 2>&1 & fi
  pid12=$!
  echo "started PID $pid12 for command: $cmd12"
  echo "             run command: $cmd13"
  if $submit; then $cmd13; fi

  for p in {$pid11,$pid12}; do
    if ps -p $p > /dev/null; then
      echo "process $p is running"
    else
      echo "process $p is not running"
    fi
  done
  dt_sec=$((SECONDS - t0_sec))
  echo "STAGE 1 IS COMPLETED time for three steps: $dt_sec sec"
fi ### $stage1


if $stage1 || $stage2; then
  echo
  echo "CHECK THAT ALL 6 ARRAYS WITH GATE LIMITS ARE AVAILABLE for -d $detname and -k $dskwargs"
  cmddir="ls -ltr $dirrepo/jungfrau/block_results/"
  echo $cmddir
  $cmddir
  echo
fi ### $stage1 || $stage2


if $stage2; then
  #echo
  echo "=== STAGE 2 - evaluate per-pixel gated average, rms, min, max, status on $nrecs events in each of 3 step/gain range"
  #### Test: sbatch --ntasks-per-node 19 jungfrau_dark_proc_sbatch.sh "jungfrau_dark_proc -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 --nrecs 100 --nrecs1 0"
  #### Ex: sbatch --wait --ntasks-per-node 19 jungfrau_dark_proc_sbatch.sh "jungfrau_dark_proc -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 --nrecs 1000 --nrecs1 0"
  #cmd_sbatch="sbatch --wait --ntasks-per-node $nranks $script_dir/jungfrau_dark_proc_sbatch.sh \"$cmd20\""
  logfile="$dirrepo/jungfrau/logs/$(date +%Y)/$(date +%Y-%m-%dT%H%M%S)_log_jungfrau_dark_proc_$(whoami)_%j.log"
  file="$script_dir/jungfrau_dark_proc_sbatch.sh"
  cmd_sbatch=(sbatch --partition=milano --account=lcls:prjdat21 --export=ALL --output=$logfile --nodes=$nnodes --ntasks-per-node=$nranks "$file" "$cmd20")
  #echo "command for sbatch: $cmd20"
  echo "cmd_sbatch split arguments:"
  show_argv "${cmd_sbatch[@]}"
  if $submit; then "${cmd_sbatch[@]}"; fi
fi # $stage2


if $stage3; then
  echo
  echo "=== STAGE 3 - deploy calibration constants"
  echo "run command: $cmd30"
  if $submit; then $cmd30; fi
fi # $stage3


dt_sec=$((SECONDS - t0_sec))
echo
echo "TOTAL CONSUMED TIME: $dt_sec sec"

#} # for timing

exit 0
