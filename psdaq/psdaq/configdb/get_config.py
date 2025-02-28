import psdaq.configdb.configdb as cdb
import json

#import pprint

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

def update_config(src, dst, verbose=False, pfx=None):
    for k, v in src.items():
        if k not in dst:
            if verbose:
                print(f'key {k} not found in dst dict')
            continue
        if isinstance(v, dict):
            v = update_config(v, dst[k], verbose, k if pfx is None else f'{pfx}.{k}')
        if (pfx is not None and pfx.startswith(':types:')) or k.endswith(':RO'):
            # Keep the value that is in dst
            if verbose and v != dst[k]:
                print('Changed:', f'{k}:' if pfx is None else f'{pfx}.{k}:', f'{v} -> {dst[k]}')
        else:
            # Keep the value that is in src
            if verbose and v != dst[k]:
                print('Changed:', f'{k}:' if pfx is None else f'{pfx}.{k}:', f'{dst[k]} -> {v}')
            dst[k] = v
    return dst

# this interface requires the detector segment
def get_config(connect_json,cfgtype,detname,detsegm):

    connect_info = json.loads(connect_json)
    control_info = connect_info['body']['control']['0']['control_info']
    instrument = control_info['instrument']
    cfg_dbase = control_info['cfg_dbase'].rsplit('/', 1)
    db_url = cfg_dbase[0]
    db_name =cfg_dbase[1]
    return get_config_with_params(db_url, instrument, db_name, cfgtype, detname+'_%d'%detsegm)

def get_config_with_params(db_url, instrument, db_name, cfgtype, detname):
    create = False
    mycdb = cdb.configdb(db_url, instrument, create, db_name)
    cfg = mycdb.get_configuration(cfgtype, detname)

    if cfg is None: raise ValueError('Config for instrument/detname %s/%s not found. dbase url: %s, db_name: %s, config_style: %s'%(instrument,detname,db_url,db_name,cfgtype))

    if '_cfgTypeRef' in cfg.keys():

        if cfg['_cfgTypeRef'] == cfgtype:
            raise ValueError('A configuration cannot be self-relative: _cfgTypeRef is %s'%(cfgtype))

        # Replace values (and add k,v) in the referenced config with those of the requested config
        # and return the combined config
        ref = mycdb.get_configuration(cfg['_cfgTypeRef'], detname)
        if ref is None:
            raise ValueError('Reference config for instrument/detname %s/%s not found. dbase url: %s, db_name: %s, config_style: %s'%(instrument,detname,db_url,db_name,cfg['_cfgTypeRef']))
        if cfg['alg:RO']['version:RO'] != ref['alg:RO']['version:RO']: # Require the version numbers to be the same
            raise ValueError('%s and %s configs for instrument/detname %s/%s must have matching alg version numbers: got %s vs %s'%
                             (cfgtype, cfg['_cfgTypeRef'], instrument, detname,
                              cfg['alg:RO']['version:RO'], ref['alg:RO']['version:RO']))
        cfg = update_config(cfg, ref)
        #print('*** final config')
        #pp = pprint.PrettyPrinter()
        #pp.pprint(cfg)
        #print('*** end')

    cfg_no_RO_names = remove_read_only(cfg)

    return cfg_no_RO_names

def get_config_json(*args):
    return json.dumps(get_config(*args))

def get_config_json_with_params(*args):
    return json.dumps(get_config_with_params(*args))
