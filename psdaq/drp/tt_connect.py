from p4p.client.thread import Context
import rogue
import lcls2_timetool

import json
import time
import pprint

def tt_connect(txId, pcidev='/dev/datadev_0'):

    myargs = { 'dev'         : pcidev,
               'pgp3'        : False,
               'pollEn'      : False,
               'initRead'    : True,
               'dataCapture' : False,
               'dataDebug'   : False,}

    # in older versions we didn't have to use the "with" statement
    # but now the register accesses don't seem to work without it -cpo
    with lcls2_timetool.TimeToolKcu1500Root(**myargs) as cl:

        rxId = cl.TimeToolKcu1500.Kcu1500Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.RxId.get()
        cl.TimeToolKcu1500.Kcu1500Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.TxId.put(txId)
        
    d = {}
    d['paddr'] = rxId
    return json.dumps(d)
