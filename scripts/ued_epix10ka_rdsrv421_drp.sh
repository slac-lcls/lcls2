#!/bin/bash

set -e

source /sdf/home/m/monarin/lcls2_worktree/ued-epix10ka-rdsrv421-daq/setup_env.sh >/dev/null
export QT_QPA_PLATFORM=offscreen

mkdir -p /tmp/prom/ued /tmp/ued_epix10ka_data/ued

exec taskset 0xfffefffe drp "$@"
