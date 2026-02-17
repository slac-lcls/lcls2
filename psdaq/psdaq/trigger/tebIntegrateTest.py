from psdaq.trigger import tebTrigger
from BldTebData import BldTebData, GasDetTebData
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
#cube_bins = 256
#cube_record_factor = cube_bins*40+1
cube_bins = 1
cube_record_factor = cube_bins*10240+1
cube_event_count = 0

def cb(ds):
    logging.warning(f'[Python] Setting nbins {cube_bins}')
    ds.cubeConfigure(cube_bins)

ds = tebTrigger.CubeTriggerDataSource(cube_bins)

connect_info = json.loads(ds.connect_json)

mebs = 0
if 'meb' in connect_info['body'].keys():
    for nodes in connect_info['body']['meb'].values():
        mebs |= 1 << nodes['meb_id']

#
#  Need to discover the timing node ID
#
drp_ids = dict()
drps    = dict()
for nodes in connect_info['body']['drp'].values():
    # For doing "ctrb.xtc.src.value() == <detector>_id" conditionals
    drp_ids[nodes['proc_info']['alias']] = nodes['drp_id']

    # For doing "drps[ctrb.xtc.src.value()] == '<detector>'" conditionals
    drps[nodes['drp_id']] = nodes['proc_info']['alias']

logging.warning(f'[Python] drp_ids {drp_ids}')

# For doing "ctrb.xtc.src.value() == <detector>_id" conditionals
timing_id  = drp_ids["timing_0"]      if "timing_0"      in drp_ids else -1
bld_id     = drp_ids["bld_0"]         if "bld_0"         in drp_ids else -1
logging.warning(f'[Python] timing_id {timing_id}  bld_id {bld_id}')

num_event = 0

#
#  The event loop
#
for event in ds.events():
    num_event += 1

    persist = False
    flush   = False
    index   = 0

    for ctrb in event:

        payld = ctrb.xtc.payload()

        if ctrb.xtc.src.value() == timing_id:
            data = TimingTebData(payld)

            if data.has_eventcode(record_eventcode):
#                persist = True
#                index = data.eventcodes_to_int(257,264)
#                cube_event_count += 1
                pass

        elif ctrb.xtc.src.value() == bld_id:
            data = BldTebData(payld)
            def test(det,parm):
                d = getattr(data,det)()
                if d is None:
                    print(f'{det} is None')
                else:
                    print(f'Found {det} with {parm}={getattr(d,parm)()}')
#            test('gmd'   ,'milliJoulesPerPulse')
#            test('xgmd'  ,'POSY')
#            test('pcav'  ,'fitTime1')
#            test('pcavs' ,'fitTime1')
#            test('ebeam' ,'l3Energy')
#            test('ebeams','l3Energy')
#            test('gasdet','f11ENRC')
            gasdet = data.gasdet()
            if gasdet is not None:
                persist = True
                index = int(gasdet.f11ENRC()) % cube_bins
                cube_event_count += 1

        else:
            data = TmoTebData(payld)

#    monitor   = (cube_event_count % cube_record_factor)<2
    recordBin = (cube_event_count % cube_record_factor)==0
    monitorBin = recordBin
    monitor = recordBin
    flush = recordBin
    ds.result(persist, mebs if monitor else 0, index, 
              recordBin, monitorBin, flush)

print(f"[Python] tebCubeTest script exiting; {num_event} events handled")
