from psalg.configdb.get_config import get_config
from p4p.client.thread import Context
import json
import time

def epics_names_values(pvtable,cfg,names,values):
    for k, v1 in pvtable.items():
        v2 = cfg[k]
        if isinstance(v1, dict):
            epics_names_values(v1,v2,names,values)
        else:
            names.append(v1)
            values.append(v2)

def hsd_config(epics_prefix,dburl,dbname,hutch,cfgtype,detname):

    cfg = get_config(dburl,dbname,hutch,cfgtype,detname)

    # this structure of epics variable names must mirror
    # the configdb.  alternatively, we could consider
    # putting these in the configdb, perhaps as readonly fields.
    pvtable = {'enable':'ENABLE',
               'raw' : {'start'    : 'RAW_START',
                        'gate'     : 'RAW_GATE',
                        'prescale' : 'RAW_PS'},
               'fex' : {'start'    : 'FEX_START',
                        'gate'     : 'FEX_GATE',
                        'prescale' : 'FEX_PS',
                        'ymin'     : 'FEX_YMIN',
                        'ymax'     : 'FEX_YMAX',
                        'xpre'     : 'FEX_XPRE',
                        'xpost'    : 'FEX_XPOST'},
               'expert' : {'datamode'   : 'TESTPATTERN',
                           'syncelo'    : 'SYNCELO',
                           'syncehi'    : 'SYNCEHI',
                           'syncolo'    : 'SYNCOLO',
                           'syncohi'    : 'SYNCOHI',
                           'fullthresh' : 'FULLEVT',
                           'fullsize'   : 'FULLSIZE',
                           'trigshift'  : 'TRIGSHIFT',
                           'pgpskip'    : 'PGPSKPINTVL'}
    }

    # this is used to know when the configuration is complete
    names = ['BASE:READY']
    values = [0]
    # look in the cfg dictionary for values that match the epics
    # variables in the pvtable
    epics_names_values(pvtable,cfg,names,values)
    names = [epics_prefix+':'+name for name in names]

    # program the values
    ctxt = Context('pva')
    myctxt = ctxt.put(names,values)
    #for name,value in zip(names,values):
    #    print('***',name,value)
    #    ctxt.put(name,value)

    # the completion of the "put" guarantees that all of the above
    # have completed (although in no particular order)
    ctxt.put(epics_prefix+':BASE:APPLYCONFIG',1)
    complete = False
    for i in range(10):
        complete = ctxt.get(names[0])==True
        if complete: break
        print('hsd config wait for complete',i)
        time.sleep(1)
    if complete:
        print('hsd config complete')
    else:
        raise Exception('timed out waiting for hsd configure')
    # turn this off so we don't reconfigure if the application is restarted
    ctxt.put(epics_prefix+':BASE:APPLYCONFIG',0)
    ctxt.close()

    return json.dumps(cfg)
