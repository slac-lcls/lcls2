from psdaq.trigger import tebTrigger
from psdaq.configdb.typed_json import cdict
from BldTebData import BldTebData, GasDetTebData, GmdTebData, XGmdTebData, PhaseCavityTebData, EBeamTebData
from HrEncoderTebData import HrEncoderTebData
from TimingTebData import TimingTebData
from TmoTebData import TmoTebData
import json
import logging

logging.warning(f"[Python] tebIntegrateTest script starting")

config = cdict()
config.setAlg('config', [2,0,0])
config.setInfo('cubeinfo','cubeinfo',0,'serial1234',__file__)
config.set('bins'         , 1          , 'UINT32') # mandatory field
d = config.typed_json()

record_eventcode = 256
#bin_eventcodes = 257..264
#cube_bins = 256
#cube_record_factor = cube_bins*40+1
cube_bins = 1
cube_record_factor = cube_bins*10240+1
cube_event_count = 0

#  This is the data source for events
ds = tebTrigger.CubeTriggerDataSource(d)

#  Lookup the detectors to use for trigger/binning decisions
timing = ds.detector('timing_0',TimingTebData)
bld    = ds.detector('bld_0'   ,BldTebData)

#  Lookup the monitoring nodes for directing monitoring events
mebs   = ds.mebs()   #  no argument results in all monitors

num_event = 0

#
#  The event loop
#
for event in ds.events():
    num_event += 1

    persist = True
    flush   = False
    index   = 0

    #  Example code for using event codes
    #    data = timing.trigger(event)
    #    if data is not None and data.has_eventcode(record_eventcode):
    #                flush = True

    #
    #  Static test
    #
    cube_event_count += 1
    recordBin = (cube_event_count % cube_record_factor)==0

    #
    #  Create the result
    #
    recordEvt = True
    recordBin = (cube_event_count % cube_record_factor)==0
    monitorBin = recordBin
    monitor = recordBin
    flush = recordBin
    ds.result(persist,      # add into sums, record single shot detectors
              recordEvt,    # record single shot data for summed detectors
              mebs if monitor else 0, # forward single shot data to monitoring
              index,        # the cube bin number (always 0)
              recordBin,    # record the current sums
              monitorBin,   # forward the current sums to monitoring
              flush)        # clear the sums after this event

print(f"[Python] tebIntegrateTest script exiting; {num_event} events handled")
