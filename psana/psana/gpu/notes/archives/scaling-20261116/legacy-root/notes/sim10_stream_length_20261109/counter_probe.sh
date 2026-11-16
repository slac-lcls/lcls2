#!/bin/bash
# Read-only discovery after all timed cases; never used to label remote bytes.
date -u
hostname
ls -ld /proc/fs/lustre/llite /sys/fs/lustre/llite /sys/kernel/debug/lustre/llite /sys/class/cxi /sys/class/cxi/cxi*/device/counters
/usr/sbin/lctl get_param 'llite.*.stats'
/usr/bin/cxi_stat --list
/usr/bin/cxi_stat --rates --pause=1
