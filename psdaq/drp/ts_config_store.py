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

top.define_enum('trigModeEnum', {'FixedRate': 0, 'ACRate': 1, 'Sequence': 2})
top.define_enum('fixedRateEnum', {'1Hz': 0, '10Hz': 1})
top.define_enum('acRateEnum', {'60Hz': 0, '30Hz': 1})
top.define_enum('boolEnum', {'False': 0, 'True': 1})
top.define_enum('seqEnum', {'Bursts': 0, '10KRates': 1, 'Local': 2})
top.define_enum('seqBurstEnum', {'2x1080ns': 0, '4x1080ns': 1})
top.define_enum('seqFixedRateEnum', {'10kHz': 0, '20kHz': 1})
top.define_enum('seqLocalEnum', {'40kHz': 0, '80kHz': 1})

for group in range(8):
    prefix = 'group'+str(group)+'.'

    top.set(prefix+'trigMode', 0, 'trigModeEnum')
    top.set(prefix+'fixed.rate', 0, 'fixedRateEnum')

    top.set(prefix+'ac.rate', 0, 'acRateEnum')
    for tsnum in range(6):
        top.set(prefix+'ac.ts'+str(tsnum), 0, 'boolEnum')

    top.set(prefix+'seq.mode', 0, 'seqEnum')
    top.set(prefix+'seq.burst.mode', 0, 'seqBurstEnum')
    top.set(prefix+'seq.fixed.rate', 0, 'seqFixedRateEnum')
    top.set(prefix+'seq.local.rate', 0, 'seqLocalEnum')

mycdb.modify_device('BEAM', top)
mycdb.print_configs()
