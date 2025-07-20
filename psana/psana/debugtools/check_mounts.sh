#!/bin/bash

NODES=(
  drp-srcf-cmp001
  drp-srcf-cmp002
  drp-srcf-cmp003
  drp-srcf-cmp004
  drp-srcf-cmp005
  drp-srcf-cmp008
  drp-srcf-cmp009
  drp-srcf-cmp010
  drp-srcf-cmp011
  drp-srcf-cmp012
  drp-srcf-cmp013
  drp-srcf-cmp014
  drp-srcf-cmp017
  drp-srcf-cmp018
  drp-srcf-cmp019
  drp-srcf-cmp020
  drp-srcf-cmp021
  drp-srcf-cmp022
  drp-srcf-cmp023
  drp-srcf-cmp024
  drp-srcf-cmp025
  drp-srcf-cmp026
  drp-srcf-cmp027
  drp-srcf-cmp028
  drp-srcf-cmp029
  drp-srcf-cmp030
  drp-srcf-cmp031
  drp-srcf-cmp032
  drp-srcf-cmp033
  drp-srcf-cmp035
  drp-srcf-cmp038
  drp-srcf-cmp039
  drp-srcf-cmp044
  drp-srcf-cmp046
  drp-srcf-cmp048
  drp-srcf-cmp050
  drp-srcf-cmp051
  drp-srcf-mon001
  drp-srcf-mon002
  drp-srcf-mon004
)

MOUNT_PATH="/cds/data/drpsrcf"
ALERT=false

LOGDIR_BASE="/cds/home/m/monarin/logs"
YEAR=$(date +%Y)
MONTH=$(date +%m)
LOGDIR="${LOGDIR_BASE}/${YEAR}/${MONTH}"
mkdir -p "$LOGDIR"

LOGFILE="${LOGDIR}/mount_check_$(date +%Y%m%d_%H%M%S).log"

echo "=== Mount Check on $(date) ===" > "$LOGFILE"

for NODE in "${NODES[@]}"; do
  timeout 10 ssh "$NODE" "mountpoint -q $MOUNT_PATH" 2>/dev/null
  if [ $? -eq 0 ]; then
    echo "[OK]   $NODE: $MOUNT_PATH is mounted" >> "$LOGFILE"
  else
    echo "[FAIL] $NODE: $MOUNT_PATH is NOT mounted or unreachable" >> "$LOGFILE"
    ALERT=true
  fi
done

echo "Mount check completed: $LOGFILE"

# Send alert if any node failed
if [ "$ALERT" = true ]; then
  mail -s "[ALERT] Mount check failure on drp-srcf nodes" monarin@slac.stanford.edu < "$LOGFILE"
fi

