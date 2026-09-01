#!/bin/bash
# code from
# https://confluence.slac.stanford.edu/spaces/~mshankar/pages/695783226/JWT+s+for+the+LCLS2+calib+service

# wrapper for getJWT.py
# grabs ENVVAL created by getJWT.py and export it to $CALIB_JWT

unset CALIB_JWT
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ENVVAL=$(${SCRIPT_DIR}/getJWT.py $@)
if [ $? -eq 0 ]
then
    export CALIB_JWT=${ENVVAL}
else
    echo "Could not get a JWT"
    exit 1
fi
