#!/usr/bin/bash
####!/usr/bin/env zsh

# lcls2/psana/psana/app/jungfrau_dark_proc_mpi.sh --skipcmds - SKIP/DO NOT execute commands of this script, just show structure of calls
# lcls2/psana/psana/app/jungfrau_dark_proc_mpi.sh -S 1 - run stage 1 ONLY !
# lcls2/psana/psana/app/jungfrau_dark_proc_mpi.sh -S 2 - run stage 2 ONLY !


# Initialize variables with default values
M14="0x3fff"
M14minus="0x3ffc"

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
version="V2026-02-10"
datbits="$M14"
deploy="False"
plotim="0"
evcode="None"
segind="None"
igmode="None"
nranks="1"   #"19" 1-no mpi
stages="ALL" # expected "1" or "2"
skipcmds=false # for debudding of this script execute/skip commands

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
            deploy="$2"
            shift
            ;;
        -p|--plotim)
            plotim="$2"
            shift
            ;;
        -c|--evcode)
            evcode="$2"
            shift
            ;;
        -I|--segind)
            segind="$2"
            shift
            ;;
        -G|--igmode)
            igmode="$2"
            shift
            ;;
        -N|--nranks)
            nranks="$2"
            shift
            ;;
        -S|--stages)
            stages="$2"
            shift
            ;;
        --skipcmds)
            skipcmds=true
            shift
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift # Skip the option argument
done


echo "skipcmds=$skipcmds"


c11="jungfrau_dark_proc -k $dskwargs -d $detname -n $nrecs --nrecs1 $nrecs1 -o $dirrepo -L $logmode -E $errskip --stepnum $stepnum --stepmax $stepmax --evskip $evskip"
c12="--dirmode $dirmode --filemode $filemode --events $events -e $evstep"
c13="--int_lo $int_lo --int_hi $int_hi --intnlo $intnlo --intnhi $intnhi --rms_lo $rms_lo --rms_hi $rms_hi --rmsnlo $rmsnlo --rmsnhi $rmsnhi"
c14="--fraclm $fraclm --fraclo $fraclo --frachi $frachi -v $version --datbits $datbits -D $deploy -p $plotim -c $evcode -I $segind -G $igmode"
c1="$c11 $c12 $c13 $c14"

#echo $c1
#cmd00="jungfrau_dark_proc -k exp=mfx100848724,run=49 -d jungfrau -o ./work1"
cmd00="jungfrau_dark_proc -k $dskwargs -d $detname -o $dirrepo -L $logmode"
cmd10="$cmd00 --nrecs $nrecs1 --nrecs1 $nrecs1 --events 10000"     #--nrecs 50 --nrecs1 50 --events 50"
cmd11="$cmd10 --stepnum 0"
cmd12="$cmd10 --stepnum 1"
cmd13="$cmd10 --stepnum 2"


#cmdmpi="mpirun --mca osc^ucx -n $nranks"
#cmdmpi="mpirun -n $nranks"

#cmdmpi=""
#if [[ "$nranks" != "1" ]]; then
#    cmdmpi="mpirun -n $nranks"
#fi

[[ "$nranks" != "1" ]] && cmdmpi="mpirun -n $nranks" || cmdmpi=""

#nevts=$(($nrecs * 3))
#echo "evaluate total number of events as $nevts"
#exit 1

cmd20="$cmdmpi $cmd00 --nrecs $nrecs --nrecs1 0 --events 10000"     #--nrecs 1000 --nrecs1 0 --events 1000"
cmd21="$cmd20 --stepnum 0"
cmd22="$cmd20 --stepnum 1"
cmd23="$cmd20 --stepnum 2"

if [[ "$logmode" == "DEBUG" ]]; then
  echo "COMMAND LINES"
  echo "common : $cmd00"
  echo "stage 1: $cmd10"
  echo "stage 2: $cmd20"
fi

t0_sec=$SECONDS

#time {

if [[ "$stages" == "1" || "$stages" == "ALL" ]]; then

echo
echo
echo "BEGIN STAGE 1 - evaluate intensity gates for $fraclo and $frachi part of statistics on $nrecs1 events"

if [[ "$skipcmds" == false ]]; then nohup $cmd11 >/dev/null 2>&1 & fi
pid11=$!
echo "started PID $pid11 for command: $cmd11"

if [[ "$skipcmds" == false ]]; then nohup $cmd12 >/dev/null 2>&1 & fi
pid12=$!
echo "started PID $pid12 for command: $cmd12"

echo "run command: $cmd13"
if [[ "$skipcmds" == false ]]; then $cmd13; fi

for p in {$pid11,$pid12}; do
  if ps -p $p > /dev/null; then
    echo "process $p is running"
  else
    echo "process $p is not running"
  fi
done

dt_sec=$((SECONDS - t0_sec))
echo "STAGE 1 IS COMPLETED time for three steps: $dt_sec sec"

fi ### "$stages" == "1" || "$stages" == "ALL"


echo
echo "CHECK THAT ALL 6 ARRAYS WITH GATE LIMITS ARE AVAILABLE for -d $detname and -k $dskwargs"
cmddir="ls -ltr $dirrepo/jungfrau/block_results/"
echo $cmddir
$cmddir
echo

if [[ "$stages" == "1" ]]; then
  echo "EXIT AFTER STAGE 1    due to -S|--stages $stages"
  exit 0
elif [[ "$stages" == "2" ]]; then
  echo "RUN STAGE 2 ONLY   due to -S|--stages $stages"
fi


echo
echo "BEGIN STAGE 2 - evaluate gated average, rms, min, max, pixel status on $nrecs events"

# === DETACHED
#nohup $cmd21 >/dev/null 2>&1 &
#pid21=$!
#echo
#echo "started PID $pid21 for command: $cmd21"

#nohup $cmd22 >/dev/null 2>&1 &
#pid22=$!
#echo "started PID $pid22 for command: $cmd22"
# ===

echo "run command: $cmd21"
if [[ "$skipcmds" == false ]]; then $cmd21; fi

echo "run command: $cmd22"
if [[ "$skipcmds" == false ]]; then $cmd22; fi

echo "run command: $cmd23"
if [[ "$skipcmds" == false ]]; then $cmd23; fi

dt_sec=$((SECONDS - t0_sec))
echo
echo "TOTAL CONSUMED TIME: $dt_sec sec"

#} # for timing

exit 0
