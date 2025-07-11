# An example showing how to run and set related parameters
# in "live" mode. You can run this example by:
# mpirun -n 3 python livemode.py
# *************************************************************


# Use environment variable to specify how many attempts,
# the datasource should wait for file reading (1 second wait).
# In this example, we set it to 30 (wait up 30 seconds).
import os
os.environ['PS_SMD_MAX_RETRIES'] = '30'


# Create a datasource with live flag
from psana2 import DataSource
ds = DataSource(exp='tmoc00118', run=222, dir='/cds/data/psdm/prj/public01/xtc', 
        live        = True,
        max_events  = 10)


# Looping over your run and events as usual
# You'll see "wait for an event..." message in case
# The file system writing is slower than your analysis
run = next(ds.runs())
for i, evt in enumerate(run.events()):
    print(f'got evt={i} ts={evt.timestamp}')
