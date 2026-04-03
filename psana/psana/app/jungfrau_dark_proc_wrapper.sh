#!/usr/bin/bash

# LOCAL PARAMETERS:
# --submit  ### for debugging, show what script is doing, but DO NOT EXECUTE COMMANDS
# --wrapper ### 0b111 - for all stages, 1-for stage 1 ONLY, 2-..., 4-...
# --slurmpars ### "--partition=milano --account=lcls:prjdat21 --export=ALL --output=2026-04-02T031738_jungfrau_dark_proc_dubrovin.log --nodes=1 --ntasks-per-node=19"

# lcls2/psana/psana/app/jungfrau_dark_proc_wrapper.sh -S 1 ### run stage 1 ONLY !
# lcls2/psana/psana/app/jungfrau_dark_proc_wrapper.sh -S 2 ### run stage 2 ONLY !
# jungfrau_dark_proc -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 --wrapper 7 [--deploy] [--submit]

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
version="V2026-04-02"
datbits="$M14"
plotim="0"
segind="None"
deploy=false
save=false
ctdepl="prs"
comment="no cmt"
run_beg="0"
run_end="end"
tstamp="None"
dbsuffix="None"
logfile="$(date +%Y-%m-%dT%H%M%S)_jungfrau_dark_proc_wrapper_$(whoami)_%j.log"
slurmpars="--partition=milano --account=lcls:prjdat21 --export=ALL --output=$logfile --nodes=1 --ntasks-per-node=5"
wrapper=7
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
        --ctdepl)
            ctdepl="$2"
            shift
            ;;
        --comment)
            comment="$2"
            shift
            ;;
        --version)
            version="$2"
            shift
            ;;
        --run_beg)
            run_beg="$2"
            shift
            ;;
        --run_end)
            run_end="$2"
            shift
            ;;
        --tstamp)
            tstamp="$2"
            shift
            ;;
        --dbsuffix)
            dbsuffix="$2"
            shift
            ;;
        -p|--plotim)
            plotim="$2"
            shift
            ;;
        -I|--segind)
            segind="$2"
            shift
            ;;
        --slurmpars)
            slurmpars="$2"
            shift
            ;;
        --wrapper)
            wrapper="$2"
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

istages=$(($wrapper))
((stage1 = istages & 1)); [[ stage1 -gt 0 ]] && stage1=true || stage1=false
((stage2 = istages & 2)); [[ stage2 -gt 0 ]] && stage2=true || stage2=false
((stage3 = istages & 4)); [[ stage3 -gt 0 ]] && stage3=true || stage3=false

#echo
echo "in wrapper: $0"
echo "     --wrapper $wrapper: do stages 1/2/3: $stage1/$stage2/$stage3"

script_dir=$(dirname "$(realpath "$0")")

cmnpars="-k $dskwargs -d $detname -o $dirrepo -L $logmode"
cmd00="jungfrau_dark_proc $cmnpars" ### "jungfrau_dark_proc -k exp=mfx100848724,run=49 -d jungfrau -o ./work1"

#time {

t0_sec=$SECONDS

cmd11="NONE"
cmd12="NONE"
cmd13="NONE"
cmd20="NONE"
cmd30="NONE"

if $stage1; then
  c01="--datbits $datbits --int_lo $int_lo --int_hi $int_hi --fraclo $fraclo --frachi $frachi"
  cmd10="$cmd00 --nrecs $nrecs1 --nrecs1 $nrecs1 $c01"
  cmd11="$cmd10 --stepnum 0"
  cmd12="$cmd10 --stepnum 1"
  cmd13="$cmd10 --stepnum 2"
fi ### $stage1

if $stage2; then
  c02_proc="--evskip $evskip" # --stepnum $stepnum --stepmax $stepmax"
  c02_status="--int_hi $int_hi --int_lo $int_lo --intnhi $intnhi --intnlo $intnlo --rms_hi $rms_hi --rms_lo $rms_lo --rmsnhi $rmsnhi --rmsnlo $rmsnlo --fraclm $fraclm"
  cmd20="$cmd00 --nrecs $nrecs --nrecs1 0 $c02_proc $c02_status"
  if $save;  then cmd20+=" --save"; fi
  #if $deploy; then cmd20+=" --deploy"; fi
fi # $stage2

if $stage3; then
  c03_pars="--ctdepl $ctdepl --version $version"
  cmd30="jungfrau_deploy_constants $cmnpars -F $c03_pars"
  if [[ "$tstamp"  != "None" ]]; then cmd30+="$ --tstamp $tstamp"; fi
  if [[ "$run_beg" != "0" ]]; then cmd30+=" --run_beg $run_beg"; fi
  if [[ "$run_end" != "end" ]]; then cmd30+=" --run_end $run_end"; fi
  if [[ "$comment" != "no cmt" ]]; then cmd30+=" --comment \"$comment\""; fi
  if [[ "$dbsuffix" != "None" ]]; then cmd30+=" --dbsuffix $dbsuffix"; fi
  if $deploy; then cmd30+=" --deploy"; fi
fi # $stage3

echo
echo "=== STAGE 1 commands - evaluate intensity gates for $fraclo and $frachi part of statistics on $nrecs1 events"
echo "cmd11: $cmd11"
echo "cmd12: $cmd12"
echo "cmd13: $cmd13"
echo
echo "=== STAGE 2 command - evaluate per-pixel gated average, rms, status[, min, max] on $nrecs events in each of 3 step/gain range"
echo "cmd20: $cmd20"
echo
echo "=== STAGE 3 command - deploy calibration constants"
echo "cmd30: $cmd30"


if $stage1 || $stage2; then
  echo
  echo "=== make sbatch list of parameters/commands"
  #### Ex: sbatch [--wait] --ntasks-per-node 19 jungfrau_dark_proc_sbatch.sh "jungfrau_dark_proc -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 --nrecs 1000 --nrecs1 0"
  file="$script_dir/jungfrau_dark_proc_sbatch.sh"
  cmd_sbatch=(sbatch)
  if $stage3; then cmd_sbatch+=(--wait); fi
  cmd_sbatch+=($slurmpars "$file" "$cmd11" "$cmd12" "$cmd13" "$cmd20" "$dirrepo")
  echo "cmd_sbatch split arguments:"
  show_argv "${cmd_sbatch[@]}"
fi # $stage1 || $stage2

if $submit; then
    "${cmd_sbatch[@]}";
else
    echo; echo "add option --submit to execute commands in sbatch"
fi

if $stage3; then
  echo
  echo "=== STAGE 3 - deploy calibration constants"
  echo "command: $cmd30"
  if $submit; then $cmd30
  else echo; echo "add option --submit to execute command above"
  fi
fi # $stage3

dt_sec=$((SECONDS - t0_sec))
echo
echo "TOTAL CONSUMED TIME: $dt_sec sec"

#} # time

exit 0
