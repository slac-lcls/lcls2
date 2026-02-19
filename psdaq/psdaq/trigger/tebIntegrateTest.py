from psdaq.trigger import tebTrigger
from BldTebData import BldTebData, GasDetTebData
from TimingTebData import TimingTebData
from TmoTebData import TmoTebData
import json
import logging

logging.warning(f"[Python] tebIntegrateTest script starting")

record_eventcode = 256
#bin_eventcodes = 257..264
#cube_bins = 256
#cube_record_factor = cube_bins*40+1
cube_bins = 1
cube_record_factor = cube_bins*10240+1
cube_event_count = 0

ds = tebTrigger.CubeTriggerDataSource(cube_bins)

timing = ds.detector('timing_0',TimingTebData)
bld    = ds.detector('bld_0'   ,BldTebData)
mebs   = ds.mebs()

num_event = 0

#
#  The event loop
#
for event in ds.events():
    num_event += 1

    persist = False
    flush   = False
    index   = 0

    data = bld.trigger(event)
    if data is not None:
        gasdet = data.gasdet()
        if gasdet is not None:
            persist = True
            index = int(gasdet.f11ENRC()) % cube_bins
            cube_event_count += 1

    recordBin = (cube_event_count % cube_record_factor)==0
    monitorBin = recordBin
    monitor = recordBin
    flush = recordBin

    ds.result(persist, mebs if monitor else 0, index, 
              recordBin, monitorBin, flush)

print(f"[Python] tebIntegrateTest script exiting; {num_event} events handled")
