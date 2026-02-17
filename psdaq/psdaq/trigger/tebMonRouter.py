from psdaq.trigger import tebTrigger
from TimingTebData import TimingTebData
from TmoTebData import TmoTebData
import json
import logging

logging.warning(f"[Python] tebMonRouter script starting")

ds = tebTrigger.TriggerDataSource()

connect_info = json.loads(ds.connect_json)

acr_eventcode = 30
ami_eventcode = 256

class MebScheduler(object):

    def __init__(self):
        self._idx = 0
        self._val = []
        self._nval = 0

    def insert(self, node):
        self._val.append(1<<node)
        self._nval += 1

    #  Attempt some round-robin here, since a partial mask 
    #  disables the teb round-robin
    def schedule(self):
        if self._nval==0:
            return 0
        idx = self._idx
        self._idx += 1
        if self._idx == self._nval:
            self._idx = 0
        return self._val[idx]

# Identify the available MEBs, if any
acr_mebs = MebScheduler()
ami_mebs = MebScheduler()
usr_mebs = MebScheduler()

if 'meb' in connect_info['body'].keys():
    for nodes in connect_info['body']['meb'].values():
        node = nodes['meb_id']
        if   'acr-meb' in nodes['proc_info']['alias']:
            acr_mebs.insert(node)
        elif 'ami-meb' in nodes['proc_info']['alias']:
            ami_mebs.insert(node)
        else:
            usr_mebs.insert(node)

logging.warning(f'[Python] MEBs:  acr {acr_mebs._val}  ami {ami_mebs._val}  usr {usr_mebs._val}')

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
    mebs    = 0

    for ctrb in event:

        if ctrb.xtc.src.value() == timing_id:

            payld = ctrb.xtc.payload()
            data = TimingTebData(payld)

            eventcodes = data.eventcodes
            logging.debug(f'[Python] Eventcodes: {eventcodes}')

            if data.has_eventcode(acr_eventcode):
                mebs = acr_mebs.schedule()
            elif data.has_eventcode(ami_eventcode):
                mebs = ami_mebs.schedule()
            else:
                mebs = usr_mebs.schedule()

            break

    ds.result(persist, mebs)

print(f"[Python] tebMonRouter script exiting; {num_event} events handled")
