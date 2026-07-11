#!/bin/bash
# Sample IB port_rcv_data (units of 4-byte words) at 1 Hz.
# On a single node all inter-rank MPI is shared-memory, so IB rx ~= FFB storage reads.
# Emits: epoch_sec  total_rcv_words  delta_words  GBps_since_last
prev=""
prevt=""
while true; do
  t=$(date +%s.%N)
  tot=0
  for f in /sys/class/infiniband/*/ports/*/counters/port_rcv_data; do
    v=$(cat "$f" 2>/dev/null || echo 0)
    tot=$((tot + v))
  done
  if [ -n "$prev" ]; then
    dt=$(awk "BEGIN{print $t-$prevt}")
    dwords=$((tot - prev))
    gbps=$(awk "BEGIN{printf \"%.3f\", ($dwords*4.0)/1e9/$dt}")
    printf "%s %s %s %s\n" "$t" "$tot" "$dwords" "$gbps"
  else
    printf "%s %s 0 0.000\n" "$t" "$tot"
  fi
  prev=$tot
  prevt=$t
  sleep 1
done
