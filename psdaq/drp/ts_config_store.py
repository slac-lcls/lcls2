from psalg.configdb.typed_json import cdict
import psalg.configdb.configdb as cdb
import os
import io

# these are the current default values, but I put them here to be explicit
create = False
dbname = 'configDB'
instrument = 'TMO'

mycdb = cdb.configdb('mcbrowne:psana@psdb-dev:9306', instrument, create, dbname)

# this needs to be called once per detType at the
# "beginning of time" to create the collection name (same as detType
# in top.setInfo).  It doesn't hurt to call it again if the collection
# already exists, however.
mycdb.add_device_config('ts')

top = cdict()

top.setInfo('ts', 'xppts', 'serial1234', 'No comment')
top.setAlg('tsConfig', [0,0,1])

# this should be an array of enums, but not yet supported
top.set('temp', 1, 'UINT32')

mycdb.modify_device('BEAM', top)
mycdb.print_configs()
