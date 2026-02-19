from psdaq.trigger import tebTrigger
from BldTebData import BldTebData, GasDetTebData, GmdTebData, XGmdTebData, PhaseCavityTebData, EBeamTebData
from HrEncoderTebData import HrEncoderTebData
from TimingTebData import TimingTebData
from TmoTebData import TmoTebData
import json
import logging

logging.warning(f"[Python] tebCubeTest script starting")

#  Test filtering on only eventcodes
#  Results are persist(T/F), bin index, monitor(T/F), record_bin(T/F)
#  persist = presence of eventcode 256
#  bin_index = bin_eventcodes
#  bin_record = (cube_event_count+1) % record_factor == 0
record_eventcode = 256
#bin_eventcodes = 257..264
cube_bins = 256
cube_record_factor = cube_bins*40+1
cube_event_count = 0

ds = tebTrigger.CubeTriggerDataSource(cube_bins)

timing = ds.detector('timing_0',TimingTebData)
bld    = ds.detector('bld_0'   ,BldTebData)
enco   = ds.detector('mono_hrencoder_0',HrEncoderTebData)
mebs   = ds.mebs()

num_event = 0

#
#  The event loop
#
for event in ds.events():
    num_event += 1

    persist = False
    index   = 0

    #    data = timing.trigger(event)
    #    if data is not None and data.has_eventcode(record_eventcode):
    #                persist = True
    #                index = data.eventcodes_to_int(257,264)
    #                cube_event_count += 1

    data = bld.trigger(event)
    if data is not None:
        gasdet = data.gasdet()
        if gasdet is not None:
            persist = True
            index = int(gasdet.f11ENRC()) % cube_bins
            cube_event_count += 1

    data = enco.trigger(event)
    if data is not None and (cube_event_count % 10000)==0:
        print(f'encoder pos{data.position()}  errCnt {data.encErrCnt()}  missTrCnt {data.missedTrigCnt()}  latches {data.latches()}')

#    monitor   = (cube_event_count % cube_record_factor)<2
    recordBin = (cube_event_count % cube_record_factor)==2
    monitorBin = recordBin
    monitor = recordBin
    ds.result(persist, mebs if monitor else 0, index, 
              recordBin, monitorBin)

print(f"[Python] tebCubeTest script exiting; {num_event} events handled")
