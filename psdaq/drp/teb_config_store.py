from psalg.configdb.typed_json import cdict
import psalg.configdb.configdb as cdb
import os
import io

# these are the current default values, but put them here to be explicit
create = False
dbname = 'configDB'
instrument = 'TMO'

mycdb = cdb.configdb('mcbrowne:psana@psdb-dev:9306', instrument, create, dbname)

# this needs to be called once per detType at the
# "beginning of time" to create the collection name (same as detType
# in top.setInfo).  It doesn't hurt to call it again if the collection
# already exists, however.
mycdb.add_device_config('teb')

top = cdict()

top.setInfo('teb', 'tmoteb', 'No serial number', 'No comment')
top.setAlg('tebConfig', [0,0,1])

top.set('soname', 'libtmoteb.so', 'CHARSTR')

mycdb.modify_device('BEAM', top)
mycdb.print_configs()
