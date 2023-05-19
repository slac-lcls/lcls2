#!/bin/bash

set -e

#while true
#do
#  echo "ci-debug branch: checking calibdb access"
#  time curl -s "https://pswww.slac.stanford.edu/calib_ws/cdb_ueddaq02/gridfs/6035d64545db0b188f7c78e8" | wc
#  echo "done checking calibdb access"
#done
python -u -c "while 1: import requests; print('*** requesting'); requests.get('https://pswww.slac.stanford.edu/calib_ws/cdb_ueddaq02/gridfs/6035d64545db0b188f7c78e8',None,timeout=60); print('*** done fetch')"
