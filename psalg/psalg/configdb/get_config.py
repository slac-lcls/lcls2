import json
import sys

import psalg.configdb.configdb as cdb

def remove_read_only(cfg):
    # be careful here: iterating recursively over dictionaries
    # while deleting items can produce strange effects (which we
    # need to do to effectively "rename" the keys without ":RO"). So
    # create a new dict, unfortunately.
    new = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            v = remove_read_only(v)
        new[k.replace(':RO', '')] = v
    return new

def get_config(dburl,dbname,hutch,cfgtype,detname):

    create = False
    mycdb = cdb.configdb(dburl, hutch, create, dbname)
    cfg = mycdb.get_configuration(cfgtype, detname)
    from bson.json_util import dumps

    # remove the readonly flags used to hide values in the
    # graphical configuration editor
    cfg_no_RO = remove_read_only(cfg)

    return dumps(cfg_no_RO)
