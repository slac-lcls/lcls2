from p4p.client.thread import Context

import json
import time
import pprint

def hsd_connect(epics_prefix):

    # Retrieve connection information from EPICS
    # PVA Server may not be up yet, so poll
    ctxt = Context('pva')
    for i in range(5):
        values = ctxt.get(epics_prefix+':PADDR_U')
        if values!=0:
            break
        time.sleep(0.1)

    ctxt.close()

    d = {}
    d['paddr'] = values
    return json.dumps(d)
