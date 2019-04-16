import json
import sys

import psalg.configdb.configdb as cdb

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

def get_config(dburl,dbname,hutch,cfgtype,detname):

    create = False
    mycdb = cdb.configdb(dburl, hutch, create, dbname)
    cfg = mycdb.get_configuration(cfgtype, detname)
    from bson.json_util import dumps

    cfg_no_RO_names = remove_read_only(cfg)

    return dumps(cfg_no_RO_names)
