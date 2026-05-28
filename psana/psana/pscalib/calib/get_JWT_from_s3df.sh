#!/bin/bash
# code from
# https://confluence.slac.stanford.edu/spaces/~mshankar/pages/695783226/JWT+s+for+the+LCLS2+calib+service

# Source the psana environment so that we can pick up krtc
source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh

/sdf/sw/s3df-cli/bin/s3df login

# If all goes well, we should have a .s3df-access-token file in your home folder

if [[ ! -f ${HOME}/.s3df-access-token ]]
then
  echo "Cannot find .s3df-access-token in your home folder"
  exit 1
fi

export CALIB_JWT=$(cat ${HOME}/.s3df-access-token)

if [[ -z "${CALIB_JWT}" ]]
then
   echo "The JWT environment variable is not set"
   exit 1
fi

# Print the JWT in case we want to inspect it
echo "CALIB_JWT: $CALIB_JWT"
echo
echo "Successfully obtained a JWT"

# Make some calib calls using the JWT. Note, in the real world, the following call is probably an sbatch
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo
echo "test command: ${SCRIPT_DIR}/test_calibcall_JWT.py"
