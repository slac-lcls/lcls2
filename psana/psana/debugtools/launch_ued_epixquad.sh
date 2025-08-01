#!/bin/bash

set -e

DATDEV_PATH="/dev/datadev_0"
LOGFILE="$HOME/drp.log"

echo "[INFO] Running DRP setup with elevated privileges..."

sudo bash -c "
  echo '[INFO] Setting ulimit...'
  ulimit -l unlimited
  ulimit -r 99

  echo '[INFO] Checking for processes using ${DATDEV_PATH}...'
  PIDS=\$(lsof | grep '${DATDEV_PATH}' | awk '{print \$2}' | sort -u)

  if [[ -n \"\$PIDS\" ]]; then
    echo '[INFO] Found processes using ${DATDEV_PATH}: \$PIDS'
    for pid in \$PIDS; do
      echo '[INFO] Killing process \$pid'
      kill -9 \$pid || true
    done
  else
    echo '[INFO] No active processes using ${DATDEV_PATH}'
  fi

  echo '[INFO] Launching DRP process... Logging to ${LOGFILE}'
  source /cds/home/m/monarin/lcls2/setup_env.sh
  taskset 0xfffefffe drp \\
    -P ued \\
    -C drp-ued-cmp002 \\
    -o /cds/home/m/monarin/tmp/daq_data_dir \\
    -k ep_provider=sockets \\
    -d ${DATDEV_PATH} \\
    -k batching=yes \\
    -l 0x1 \\
    -D epixquad \\
    -k timebase=119M \\
    -u epixquad_0 \\
    -p 0 >> ${LOGFILE} 2>&1
"

