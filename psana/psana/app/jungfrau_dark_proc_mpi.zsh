#!/usr/bin/env zsh

# Initialize variables with default values
M14="0x3fff"
M14minus="0x3ffc"

dskwargs="exp=mfxdaq23,run=7,dir=/sdf/data/lcls/drpsrcf/ffb/MFX/mfxdaq23/xtc"
detname="jungfrau"
nrecs="1000"
nrecs1="50"
dirrepo="/sdf/group/lcls/ds/ana/detector/calib2/constants"
logmode="INFO"
errskip="True"
stepnum="None"
stepmax="3"
evskip="0"
events="1000000"
evstep="1000000"
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
version="V2026-01-13"
datbits="$M14"
deploy="False"
plotim="0"
evcode="None"
segind="None"
igmode="None"
mpi="False"

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
        -M|--mpi)
            mpi="$2"
            shift
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift # Skip the option argument
done


c11="jungfrau_dark_proc -k $dskwargs -d $detname -n $nrecs --nrecs1 $nrecs1 -o $dirrepo -L $logmode -E $errskip --stepnum $stepnum --stepmax $stepmax --evskip $evskip"
c12="--dirmode $dirmode --filemode $filemode --events $events -e $evstep"
c13="--int_lo $int_lo --int_hi $int_hi --intnlo $intnlo --intnhi $intnhi --rms_lo $rms_lo --rms_hi $rms_hi --rmsnlo $rmsnlo --rmsnhi $rmsnhi"
c14="--fraclm $fraclm --fraclo $fraclo --frachi $frachi -v $version --datbits $datbits -D $deploy -p $plotim -c $evcode -I $segind -G $igmode -M $mpi"
c1="$c11 $c12 $c13 $c14"

echo $c1
