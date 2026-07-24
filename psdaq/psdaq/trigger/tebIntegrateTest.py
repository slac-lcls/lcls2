from psdaq.trigger import tebTrigger
from psdaq.configdb.typed_json import cdict
import json
import logging

fastCamWindow = 0
slowCamWindow = 1
#  end-of-integration is group 2 and group 3 trigger
fastCamGroup = 1<<2
slowCamGroup = 1<<3

logging.warning(f"[Python] tebIntegrateTest script starting")

config = cdict()
config.setAlg('config', [2,0,0])
config.setInfo('cubeinfo','cubeinfo',0,'serial1234',__file__)
#  number of simultaneous integration windows; e.g. fast camera, slow camera
config.set('bins'         , 2          , 'UINT32') # mandatory field
d = config.typed_json()

#  This is the data source for events
ds = tebTrigger.WindowTriggerDataSource(d)

#  Lookup the monitoring nodes for directing monitoring events
mebs   = ds.mebs()   #  no argument results in all monitors

num_event = 0

#
#  The event loop
#
for event in ds.events():
    num_event += 1

    #  List of windows to add the event into
    addList     = [fastCamWindow,slowCamWindow]
    #  List of windows to close
    closeList   = []

    rog = event.readoutGroups()
    if rog & fastCamGroup:
        closeList.append(fastCamWindow)
    if rog & slowCamGroup:
        closeList.append(slowCamWindow)

    monitor   = (rog & (fastCamGroup | slowCamGroup))!=0
        
    #
    #  Create the result
    #

    ds.result(True,         # add into sums, record single shot detectors
              True,         # record single shot data for summed detectors
              mebs if monitor else 0, # forward single shot data to monitoring
              addList,      # add the event to these
              closeList,    # record the current sums
              closeList,    # forward the current sums to monitoring
              closeList)    # clear the sums after this event

print(f"[Python] tebIntegrateTest script exiting; {num_event} events handled")
