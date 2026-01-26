from psdaq.trigger import tebTrigger
from TimingTebData import TimingTebData
from TmoTebData import TmoTebData
import json
import logging

logging.warning(f"[Python] tebCubeTest script starting")

ds = tebTrigger.TriggerDataSource()

connect_info = json.loads(ds.connect_json)

#  Test filtering on only eventcodes
#  Results are persist(T/F), bin index, worker, monitor(T/F), record_bin(T/F)
#  persist = presence of eventcode 256
#  bin_index = bin_eventcodes
#  worker = bin_index % cube_workers
#  bin_record = (cube_event_count+1) % record_factor == 0
record_eventcode = 256
#bin_eventcodes = 257..264
cube_bins = 256
cube_workers = 8
cube_event_count = 0
cube_record_factor = cube_bins*17+1

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
timing_id  = drp_ids["timing_1"]      if "timing_1"      in drp_ids else -1
logging.warning(f'[Python] timing_id {timing_id}')

num_event = 0

#
#  The event loop
#
for event in ds.events():
    num_event += 1

    persist = False
    index   = 0

    for ctrb in event:

        if ctrb.xtc.src.value() == timing_id:

            payld = ctrb.xtc.payload()
            data = TimingTebData(payld)

#            logging.warning(f'Found timing with eventcodes {data.eventcodes()}')

            if data.has_eventcode(record_eventcode):
                persist = True
                index = data.eventcodes_to_int(257,264)
                cube_event_count += 1
#                logging.warning(f'[Python] event {cube_event_count}  bin {index}')
                
        else:
            payld = ctrb.xtc.payload()
            data = TmoTebData(payld)

#            logging.warning(f'[Python] write {data.write:x}')

    ds.cubeResult(persist, mebs, index, index%cube_workers, 
                  (cube_event_count % cube_record_factor)==0)

print(f"[Python] tebCubeTest script exiting; {num_event} events handled")
