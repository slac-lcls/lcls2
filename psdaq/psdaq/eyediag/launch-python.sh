#!/bin/bash
cwd=$(dirname "$0")
cd $cwd
source ../../../setup_env.sh
python $*
