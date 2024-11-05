#/bin/bash

if (( $# != 1 )); then
    >&2 echo "Usage: $0 <numberOfSeconds>"
    exit 1
fi

timed_run --rix --duration="$1"   
