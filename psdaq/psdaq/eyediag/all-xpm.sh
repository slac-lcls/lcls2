#!/bin/bash
cwd=$(dirname "$0")
cd $cwd
source ../../../setup_env.sh
python xpm.py --ip $1 --bathtub --write $2
