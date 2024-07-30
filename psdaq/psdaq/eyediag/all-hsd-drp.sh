#!/bin/bash
cwd=$(dirname "$0")
cd $cwd
source ../../../setup_env.sh
python hsd-drp.py --bathtub --dev /dev/datadev_0 --write $1
python hsd-drp.py --bathtub --dev /dev/datadev_1 --write $1
