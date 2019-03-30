import json

import psalg.configdb.configdb as cdb
create = False

dbname = 'cpotest'
mycdb = cdb.configdb('cpo:psana@psdb-dev:9306', 'AMO', create, dbname)
cfg = mycdb.get_configuration('BEAM', 'xpphsd1')
from bson.json_util import dumps

# this is a global symbol that the calling C code can lookup
# to get the json
config_json = dumps(cfg)
