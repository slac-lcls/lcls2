#!/bin/bash
set -euo pipefail
attempt=$1
code=$2
export SIM10_RUNTIME=C
# The attempt has frozen clean_exec/runtime files and the reviewed source tree.
timeout --signal=TERM --kill-after=30s 6600s /usr/bin/python3 "$code/stage_input.py" "$attempt/input-plan.json"
timeout --signal=TERM --kill-after=30s 6600s /usr/bin/python3 "$attempt/bundle/clean_exec.py" "$attempt" python "$code/qualify_data.py" "$attempt/site.json" "$attempt/qualified"
