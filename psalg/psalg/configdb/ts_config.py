from psalg.configdb.get_config import get_config
from p4p.client.thread import Context
from bson.json_util import dumps
import time

def ts_config(epics_prefix,dburl,dbname,hutch,cfgtype,detname):

    cfg = get_config(dburl,dbname,hutch,cfgtype,detname)
    return dumps(cfg)
