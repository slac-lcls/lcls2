import psalg.configdb.configdb as cdb
import json

# json2xtc conversion depends on these being present with ':RO'
# (and the :RO does not appear in the xtc names)
leave_alone = ['detName:RO','detType:RO','detId:RO','doc:RO','alg:RO','version:RO']

def remove_read_only(cfg):
    # be careful here: iterating recursively over dictionaries
    # while deleting items can produce strange effects (which we
    # need to do to effectively "rename" the keys without ":RO"). So
    # create a new dict, unfortunately.
    new = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            v = remove_read_only(v)
        if k in leave_alone:
            new[k] = v
        else:
            new[k.replace(':RO', '')] = v
    return new

def get_config(connect_json,cfgtype,detname):

    connect_info = json.loads(connect_json)
    control_info = connect_info['body']['control']['0']['control_info']
    instrument = control_info['instrument']
    cfg_dbase = control_info['cfg_dbase'].split('/')
    db_url = cfg_dbase[0]
    db_name =cfg_dbase[1]

    create = False
    mycdb = cdb.configdb(db_url, instrument, create, db_name)
    cfg = mycdb.get_configuration(cfgtype, detname)

    if cfg is None: raise ValueError('Config for instrument/detname %s/%s not found. dbase url: %s, db_name: %s, config_style: %s'%(instrument,detname,db_url,db_name,cfgtype))

    cfg_no_RO_names = remove_read_only(cfg)

    return cfg_no_RO_names

def get_config_json(*args):
    return json.dumps(get_config(*args))
