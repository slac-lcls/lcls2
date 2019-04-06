import json
import sys

import psalg.configdb.configdb as cdb

def get_config(dburl,dbname,hutch,cfgtype,detname):

    create = False
    mycdb = cdb.configdb(dburl, hutch, create, dbname)
    cfg = mycdb.get_configuration(cfgtype, detname)
    from bson.json_util import dumps

    # this is a global symbol that the calling C code can lookup
    # to get the json
    return dumps(cfg)
