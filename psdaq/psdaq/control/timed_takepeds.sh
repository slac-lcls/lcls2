#/bin/bash

#
# Example: Record a 10 second run on RIX DAQ
#
#   ./timed_takepeds.sh 10 rix.py
#

# check the number of arguments, and exit if incorrect.
if (( $# != 2 )); then
    >&2 echo "Usage: $0 <time_in_seconds> <path_to_config_file>"
    exit 1
fi

# check if the config file is found
if [ ! -f "$2" ]; then
    >&2 echo "ERROR:File not found: $2"
    exit 1
fi

# run the daq for the specified duration with recording ENABLED
timed_run --duration="$1" --config="$2" --record
