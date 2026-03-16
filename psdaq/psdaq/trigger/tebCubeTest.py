from psdaq.trigger import tebTrigger
from psdaq.configdb.typed_json import cdict
from BldTebData import BldTebData, GasDetTebData, GmdTebData, XGmdTebData, PhaseCavityTebData, EBeamTebData
from HrEncoderTebData import HrEncoderTebData
from TimingTebData import TimingTebData
from TmoTebData import TmoTebData
import json
import logging

logging.warning(f"[Python] tebCubeTest script starting")

config = cdict()
config.setAlg('config', [2,0,0])
config.setInfo('cubeinfo','cubeinfo',0,'serial1234',__file__)
config.set('bins'         , 8192       , 'UINT32')  # mandatory field - flattened
config.set('bin_dims'     , [64, 64, 2], 'UINT32')  # individual dimensions
config.set('record_code'  , 256        , 'UINT16')  # example config data
config.set('record_factor', 5121       , 'UINT32')  # --
config.set('use_encoder'  , 0          , 'UINT8')
config.set('use_gasdet'   , 0          , 'UINT8')
d = config.typed_json()

print('-- configuration --')
print(json.dumps(d))

#  This is the data source for events
ds = tebTrigger.CubeTriggerDataSource(d)

#  Lookup the detectors to use for trigger/binning decisions
timing = ds.detector('timing_0',TimingTebData)
bld    = ds.detector('bld_0'   ,BldTebData)
enco   = ds.detector('mono_hrencoder_0',HrEncoderTebData)

#  Lookup the monitoring nodes for directing monitoring events
mebs   = ds.mebs()   #  no argument results in all monitors

print(f'-- mebs {mebs:x}')

num_event = 0
cube_event_count = 0

#
#  The event loop
#
for event in ds.events():
    num_event += 1

    persist = False
    index   = 0

    #  Example code for using event codes
    #    data = timing.trigger(event)
    #    if data is not None and data.has_eventcode(record_eventcode):
    #                persist = True
    #                index = data.eventcodes_to_int(257,264)
    #                cube_event_count += 1

    #  Example code for using gas detector
    if d['use_gasdet']:
        data = bld.trigger(event)
        if data is not None:
            gasdet = data.gasdet()
            if gasdet is not None:
                persist = True
                index = int(gasdet.f11ENRC()) % d['bins']
                cube_event_count += 1

    #  Example code for using mono encoder
    if d['use_encoder']:
        data = enco.trigger(event)
        if data is not None and (cube_event_count % 10000)==0:
            print(f'encoder pos{data.position()}  errCnt {data.encErrCnt()}  missTrCnt {data.missedTrigCnt()}  latches {data.latches()}')

    #
    #  Static test assignment
    #
    if not d['use_gasdet'] and not d['use_encoder']:
        persist = True
        cube_event_count += 1
        index = cube_event_count % d['bins']

    #
    #  Create the result
    #
    recordEvt = 0   
    recordBin = (cube_event_count % d['record_factor'])==4  
    monitorBin = recordBin  
    monitor = recordBin     
    ds.result(persist,      # add into cube, record single shot detectors
              recordEvt,    # record single shot data for "cubed" detectors
              mebs if monitor else 0, # forward single shot data to monitoring
              index,        # the cube bin number
              recordBin,    # record the current cube bin
              monitorBin)   # forward the current bin to monitoring

print(f"[Python] tebCubeTest script exiting; {num_event} events handled; {cube_event_count} events accepted")
