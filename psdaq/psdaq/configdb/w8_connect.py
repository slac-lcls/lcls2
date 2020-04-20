from p4p.client.thread import Context

import json
import time
import pprint

def w8_connect(epics_prefix):

    # Retrieve connection information from EPICS
    # May need to wait for other processes here, so poll
    ctxt = Context('pva')
    for i in range(50):
        values = ctxt.get(epics_prefix+':Top:TriggerEventManager:XpmMessageAligner:RxId').raw.value
        if values!=0:
            break
        print('{:} is zero, retry'.format(epics_prefix+':Top:TriggerEventManager:XpmMessageAligner:RxId'))
        time.sleep(0.1)

    ctxt.close()

    d = {}
    d['paddr'] = values
    return json.dumps(d)
