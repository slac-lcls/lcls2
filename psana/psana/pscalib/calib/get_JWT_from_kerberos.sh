#!/bin/bash

# code from
# https://confluence.slac.stanford.edu/spaces/~mshankar/pages/695783226/JWT+s+for+the+LCLS2+calib+service

# Get a PCDS JWT using your Kerberos token
# WORKS ON INTERACTIVE NODES ONLY!!!!

source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# We will eventually get getJWT.sh and getJWT.py from the psana environment but for now stage these somewhere in your PATH
source ${SCRIPT_DIR}/getJWT.sh calib

# Print the JWT in case we want to inspect it
echo $CALIB_JWT

if [[ -z "${CALIB_JWT}" ]]
then
   echo "The JWT environment variable is not set"
   exit 1
fi

echo "Successfully obtained a JWT"
echo
echo "test command: ${SCRIPT_DIR}/test_calibcall_JWT.py"
