from psdaq.trigger import tebTrigger
from TimingTebData import TimingTebData
from TmoTebData import TmoTebData
import json
import logging

logging.warning(f"[Python] tebFilter script starting")

ds = tebTrigger.TriggerDataSource()

connect_info = json.loads(ds.connect_json)

flush_eventcode = 257

mebs = 0
if 'meb' in connect_info['body'].keys():
    for nodes in connect_info['body']['meb'].values():
        mebs |= 1 << nodes['meb_id']

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

for event in ds.events():
    num_event += 1

    persist = True

    for ctrb in event:

        if ctrb.xtc.src.value() == timing_id:

            payld = ctrb.xtc.payload()
            data = TimingTebData(payld)

            eventcodes = data.eventcodes
            logging.debug(f'[Python] Eventcodes: {eventcodes}')

            if data.has_eventcode(flush_eventcode):
                persist = False
            break    

    ds.result(persist, mebs if persist else 0)

print(f"[Python] tebMonRouter script exiting; {num_event} events handled")
