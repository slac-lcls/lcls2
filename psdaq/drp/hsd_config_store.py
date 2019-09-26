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
# this needs to be called once per detType at the
# "beginning of time" to create the collection name (same as detType
# in top.setInfo).  It doesn't hurt to call it again if the collection
# already exists, however.
mycdb.add_device_config('hsd')

top = cdict()

top.setInfo('hsd', 'xpphsd', 'serial1234', 'No comment')
top.setAlg('hsdConfig', [0,0,1])

# this should be an array of enums, but not yet supported
top.set('enable', 1*[1], 'UINT32')

top.set('raw.start', 1*[4], 'UINT32')
top.set('raw.gate', 1*[20], 'UINT32')
top.set('raw.prescale', 1*[1], 'UINT32')

top.set('fex.start', 1*[4], 'UINT32')
top.set('fex.gate', 1*[20], 'UINT32')
top.set('fex.prescale', 1*[1], 'UINT32')
top.set('fex.ymin', 1*[0], 'UINT32')
top.set('fex.ymax', 1*[2000], 'UINT32')
top.set('fex.xpre', 1*[2], 'UINT32')
top.set('fex.xpost', 1*[1], 'UINT32')

top.define_enum('dataModeEnum', {'Data': -1, 'Ramp': 0, 'Spike11': 1, 'Spike12': 3, 'Spike16': 5})

top.set('expert.datamode', -1, 'dataModeEnum')
top.set('expert.fullthresh', 4, 'UINT32')
top.set('expert.fullsize', 3072, 'UINT32')
top.set('expert.fsrange', 65535, 'UINT32')
top.set('expert.trigshift', 0, 'UINT32')
top.set('expert.synce', 0, 'UINT32')
top.set('expert.synco', 0, 'UINT32')

mycdb.modify_device('BEAM', top)
mycdb.print_configs()
