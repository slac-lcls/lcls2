from psalg.configdb.get_config import get_config
from p4p.client.thread import Context
from bson.json_util import dumps

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

    names = []
    values = []
    # look in the cfg dictionary for values that match the epics
    # variables in the pvtable
    epics_names_values(pvtable,cfg,names,values)
    names = [epics_prefix+':'+name for name in names]
    names.append(epics_prefix+':BASE:APPLYCONFIG')
    values.append(1)

    # program the values
    ctxt = Context('pva')
    myctxt = ctxt.put(names,values)
    #for name,value in zip(names,values):
    #    print('***',name,value)
    #    ctxt.put(name,value)
    ctxt.close()

    return dumps(cfg)
