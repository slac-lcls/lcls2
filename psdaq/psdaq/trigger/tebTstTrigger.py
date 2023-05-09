from psdaq.trigger import tebTrigger
from TmoTebData    import TmoTebData

from psdaq.configdb.get_config import *

import json

transition_id = {
        0: "ClearReadout",
        1: "Reset",
        2: "Configure",
        3: "Unconfigure",
        4: "BeginRun",
        5: "EndRun",
        6: "BeginStep",
        7: "EndStep",
        8: "Enable",
        9: "Disable",
        10: "SlowUpdate",
        11: "Unused_11",
        12: "L1Accept",
        13: "NumberOf",
}

ALL_MEBS = 0xffffffff
NO_MEBS  = 0


print(f"[Python] tebTstTrigger script starting")

cfgtype = 'BEAM'
detname = 'trigger'
detsegm = 0

ds = tebTrigger.TriggerDataSource()

print(f"DEBUG:",ds.connect_json) # DEBUG

connect_info = json.loads(ds.connect_json)

# Identify the available MEBs, if any
ami_meb = ALL_MEBS
usr_meb = 0
mebs = dict()
if 'meb' in connect_info['body'].keys():
    for nodes in connect_info['body']['meb'].values():
        if   nodes['proc_info']['alias'] == 'ami-meb0':
            ami_meb  = 1 << nodes['meb_id']
        else:
            usr_meb |= 1 << nodes['meb_id']
if usr_meb == 0:
    usr_meb = ALL_MEBS

# Check for 'slow' readout groups
slowRogs = 0
for nodes in connect_info['body']['drp'].values():
    rog = nodes['det_info']['readout']
    if rog != ds.args.p:
        slowRogs |= 1 << rog

# Distribute events amongst all MEBs when there are no 'slow' readout groups
if slowRogs == 0:
    usr_meb |= ami_meb

# Prepare to analyze different detectors differently
drp_ids = dict()
drps    = dict()
for nodes in connect_info['body']['drp'].values():
    # For doing "ctrb.xtc.src.value() == <detector>_id" conditionals
    drp_ids[nodes['proc_info']['alias']] = nodes['drp_id']

    # For doing "drps[ctrb.xtc.src.value()] == '<detector>'" conditionals
    drps[nodes['drp_id']] = nodes['proc_info']['alias']

# For doing "ctrb.xtc.src.value() == <detector>_id" conditionals
timing_id  = drp_ids["timing_0"]      if "timing_0"      in drp_ids else -1
bld_id     = drp_ids["bld_0"]         if "bld_0"         in drp_ids else -1
epics_id   = drp_ids["epics_0"]       if "epics_0"       in drp_ids else -1
piranha_id = drp_ids["tstpiranha4_0"] if "tstpiranha4_0" in drp_ids else -1
wave8_id   = drp_ids["tst_fim0_0"]    if "tst_fim0_0"    in drp_ids else -1
tstcam1_id = drp_ids["tstcam1_0"]     if "tstcam1_0"     in drp_ids else -1
tstcam2_id = drp_ids["tstcam2_0"]     if "tstcam2_0"     in drp_ids else -1
manta_id   = drp_ids["manta_0"]       if "manta_0"       in drp_ids else -1
hsd10_id   = drp_ids["hsd_10"]        if "hsd_10"        in drp_ids else -1
hsd11_id   = drp_ids["hsd_11"]        if "hsd_11"        in drp_ids else -1

cfg = get_config(ds.connect_json, cfgtype, detname, detsegm)

persistValue = cfg["persistValue"]
monitorValue = cfg["monitorValue"]

num_event = 0

for event in ds.events():
    num_event += 1

    persist = False
    monitor = False

    for ctrb in event:

        timestamp_sec = ctrb.time.seconds()
        timestamp_ns  = ctrb.time.nanoseconds()
        ctrb_transition_id = transition_id[ctrb.service()]

        #print(f"[Python] Ctrb - "
        #      f"PulseId: {'%014x' % ctrb.pulseId()}, "
        #      f"Timestamp: {timestamp_sec}.{'%09u' % timestamp_ns}, "
        #      f"Env: {'%08x' % ctrb.env}, "
        #      f"Service: {ctrb.service()} ({ctrb_transition_id}), "
        #      f"Src: {ctrb.xtc.src.value()}, "
        #      f"Damage: {ctrb.xtc.damage.value()}, "
        #      f"TypeId: {ctrb.xtc.contains.value()}, "
        #      f"Extent: {ctrb.xtc.extent}, "
        #      f"PayloadSize: {ctrb.xtc.sizeofPayload()}")

        #src = ctrb.xtc.src.value()
        #print(f"Saw ctrb[{src}] = {drps[src]}")

        data = TmoTebData(ctrb.xtc.payload())

        #print(f"[Python] Payload - "
        #      f"write: {'%08x' % data.write}, "
        #      f"monitor: {'%08x' % data.monitor}")

        if data.write   == persistValue:  persist = True
        if data.monitor == monitorValue:  monitor = True

    # For a test, send only 'slow' readout group events to the AMI MEB
    # Same answer for all contributions, so outside the contribution loop
    # When there are no slow readout groups, monitored events go to all MEBs
    mebs = ami_meb if ctrb.readoutGroups() & slowRogs else usr_meb

    ds.result(persist, mebs if monitor else NO_MEBS)

print(f"[Python] tebTstTrigger script exiting; {num_event} events handled")
