#!/usr/bin/bash

# condaProcServ.sh
#
# arguments:
#   1: condaEnv
#   2: name
#   3: waitFlag
#   4: logFile
#   5: coreSize
#   6: ctrlPort
#   7: pythonVer
#   8: cmd
#   9+: optional arguments to cmd
#
# return values:
#   0: success
#   5: invalid arguments
#   6: conda activate failed
#   7: command not found on PATH
#   other: procServ error

if [ "$#" -lt 8 ]; then
    echo "usage: $0 <condaEnv> <name> <waitFlag> <logFile> <coreSize> <ctrlPort> <pythonVer> <cmd> [<arg1> ...]"
    exit 5
fi

# start with minimal PATH
export PATH="/usr/sbin:/usr/bin:/sbin:/bin"

source /reg/g/psdm/sw/conda2/inst/etc/profile.d/conda.sh

condaEnv="$1"
name="$2"
waitFlag="$3"
logFile="$4"
coreSize="$5"
ctrlPort="$6"
# remove .* suffix from $7, shortening xx.yy.zz to xx.yy
pythonVer="${7%.*}"
cmd="$8"
args=""
if [ "$#" -gt 8 ]; then
    shift 8
    args="$@"
fi

if [ -z $condaEnv ]; then
    echo "no conda environment specified"
elif ! conda activate "$condaEnv"; then
    echo "error: conda activate $condaEnv"
    exit 6
fi

# if TESTRELDIR has been set, update paths
if [[ -v TESTRELDIR ]]; then
    # TESTRELDIR before condaEnv in PATH
    export PATH="${TESTRELDIR}/bin:${PATH}"
    # SET LD_LIBRARY_PATH
    export LD_LIBRARY_PATH="${TESTRELDIR}/lib"
    if [ ! -z $pythonVer ]; then
        # SET PYTHONPATH
        export PYTHONPATH="${TESTRELDIR}/lib/python${pythonVer}/site-packages"
    fi
else
    echo "TESTRELDIR has not been set"
fi

# expand command to absolute path
fullpath=`type -P "$cmd"`
if [ -z $fullpath ]; then
    echo "error: not found on PATH: '$cmd'"
    exit 7
else
    cmd=$fullpath
fi

if [ -z $logFile ]; then
    logFlag=""
else
    logFlag="-L $logFile"
fi

if [ -z $waitFlag ]; then
    waitFlag=""
fi

/reg/common/package/procServ/2.6.0-SLAC/x86_64-rhel6-gcc44-opt/bin/procServ \
    --noautorestart --name $name $waitFlag $logFlag --allow \
    --coresize $coreSize $ctrlPort $cmd $args

rv=$?

if [ "$rv" -ne 0 ]; then
    echo "error: procServ returned $rv"
fi

exit $rv
