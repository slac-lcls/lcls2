from psalg.configdb.typed_json import cdict
import psalg.configdb.configdb as cdb
import os
import io

# these are the current default values, but I put them here to be explicit
create = False
dbname = 'configDB'
instrument = 'TMO'

mycdb = cdb.configdb('mcbrowne:psana@psdb-dev:9306', instrument, create, dbname)
print('Configs:')
mycdb.print_configs()
print(70*'-')
# this needs to be called once to create the
# collection name (same as detType in top.setInfo)
mycdb.add_device_config('hsd')

top = cdict()

top.setInfo('hsd', 'xpphsd', 'serial1234', 'No comment')
top.setAlg('hsdConfig', [0,0,1])
# Scalars
top.define_enum('EnableEnum', {'Disable': 0, 'Enable': 1})
#top.set('top.enable', 1, 'EnableEnum')
top.set('enable', 1, 'EnableEnum')

top.set('raw.start', 4, 'UINT16')
top.set('raw.gate', 20, 'UINT16')
top.set('raw.prescale', 1, 'UINT16')

top.set('fex.start', 4, 'UINT16')
top.set('fex.gate', 20, 'UINT16')
top.set('fex.prescale', 1, 'UINT16')
top.set('fex.ymin', 0, 'UINT16')
top.set('fex.ymax', 2000, 'UINT16')
top.set('fex.xpre', 2, 'UINT16')
top.set('fex.xpost', 1, 'UINT16')

mycdb.modify_device('BEAM', top)
mycdb.print_configs()
