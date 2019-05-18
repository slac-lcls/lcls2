from psalg.configdb.get_config import get_config
from p4p.client.thread import Context
from bson.json_util import dumps
import time

def ts_config(epics_prefix,dburl,dbname,hutch,cfgtype,detname):

    cfg = get_config(dburl,dbname,hutch,cfgtype,detname)

    # get the base from the collection?
    base = 'DAQ:LAB2:PART:4'

    # ClearReadout increments the XPM event counter, which the
    # PGP event builder needs to start at 0.  So it's critical
    # "clear readout" happen before the "clear" command, which
    # resets the event counter.
    clear_name = base+':'+'MsgClear'
    names = 3*[clear_name]
    values = [0,1,0]

    # do all of these serially. might be able to do some in parallel
    # to reduce time
    ctxt = Context('pva')
    for name,value in zip(names,values):
        print('***',name,value)
        ctxt.put(name,value)

    ctxt.close()

    return dumps(cfg)
