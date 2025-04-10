import json

import epics

from psdaq.configdb.get_config import get_config


def pvadetector_config(connect_str,cfgtype,detname,detsegm):
    #  Read the configdb
    try:
        cfg = get_config(connect_str,cfgtype,detname,detsegm)
    except Exception as err:
        print(err)
        return None
    modifiedCfg = cfg.copy()
    # NOTE: Probably don't need casing on channel access. epics.PV.get/put falls
    # back as needed. Will leave for now in case use cases arise.
    for key in cfg:
        if key == "read_only":
            # Read-only processing: replace read_only.desc.value with pvget/caget value
            for pvDesc in cfg[key]:
                if cfg[key][pvDesc]["ca"] == 1:
                    modifiedCfg[key][pvDesc]["value"] = epics.caget(cfg[key][pvDesc]["pvname"])
                else:
                    modifiedCfg[key][pvDesc]["value"] = epics.PV(cfg[key][pvDesc]["pvname"]).get()
        elif key == "write":
            # Write processing: put write.desc.value to the corresponding pv
            for pvDesc in cfg[key]:
                if cfg[key][pvDesc]["ca"] == 1:
                    ret = epics.caput(cfg[key][pvDesc]["pvname"], cfg[key][pvDesc]["value"])
                    # Store actual value IOC uses afterwards
                    if ret:
                        modifiedCfg[key][pvDesc]["value"] = epics.caget(cfg[key][pvDesc]["pvname"])
                else:
                    ret = epics.PV(cfg[key][pvDesc]["pvname"]).put(cfg[key][pvDesc]["value"])
                    # Store actual value IOC uses afterwards
                    if ret:
                        modifiedCfg[key][pvDesc]["value"] = epics.PV(cfg[key][pvDesc]["pvname"]).get()
    return json.dumps(modifiedCfg)
