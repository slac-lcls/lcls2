from p4p.client.thread import Context

import json
import time
import pprint

def hsd_connect(epics_prefix):

    # Retrieve connection information from EPICS
    ctxt = Context('pva')
    values = ctxt.get(epics_prefix+':PADDR_U')
    print(values)

    ctxt.close()

    d = {}
    d['paddr'] = values
    return json.dumps(d)
