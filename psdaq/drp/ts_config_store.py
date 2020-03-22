from psalg.configdb.typed_json import cdict
import psalg.configdb.configdb as cdb

# these are the current default values, but I put them here to be explicit
create = True
dbname = 'configDB'
instrument = 'tst'

mycdb = cdb.configdb('mcbrowne:psana@psdb-dev:9306', instrument, create, dbname)
mycdb.add_alias("BEAM")

# this needs to be called once per detType at the
# "beginning of time" to create the collection name (same as detType
# in top.setInfo).  It doesn't hurt to call it again if the collection
# already exists, however.
mycdb.add_device_config('ts')

top = cdict()

top.setInfo('ts', 'tmots', 'serial1234', 'No comment')
top.setAlg('tsConfig', [0,1,1])

top.define_enum('trigModeEnum', {key:val for val,key in enumerate(
    ['FixedRate', 'ACRate', 'Sequence', 'EventCode'])})
top.define_enum('fixedRateEnum', {key:val for val,key in enumerate(
    ['929kHz', '71_4kHz', '10_2kHz', '1_02kHz', '102Hz', '10_2Hz', '1_02Hz'])})
top.define_enum('acRateEnum', {key:val for val,key in enumerate(
    ['60Hz', '30Hz', '10Hz', '5Hz', '1Hz'])})
top.define_enum('boolEnum', {'False':0, 'True':1})
top.define_enum('seqEnum', {'Bursts': 15, '10KRates': 16, 'Local': 17})

seqBurstNames = []
for j in range(16):
    seqBurstNames.append('%dx%dns'%(2**(1+(j%4)),1080*(j/4)))

top.define_enum('seqBurstEnum', {key:val for val,key in enumerate(seqBurstNames)})

seqFixedRateNames = []
for j in range(16):
    seqFixedRateNames.append('%dkHz'%((j+1)*10))

top.define_enum('seqFixedRateEnum', {key:val for val,key in enumerate(seqFixedRateNames)})

seqLocal = ['%u0kHz'%(4*i+4) for i in range(16)]
top.define_enum('seqLocalEnum', {key:val for val,key in enumerate(seqLocal)})
top.define_enum('destSelectEnum', {'Include': 0, 'DontCare': 1})

for group in range(8):
    grp_prefix = 'group'+str(group)+'.'

    top.set(grp_prefix+'trigMode', 0, 'trigModeEnum') # default to fixed rate
    top.set(grp_prefix+'delay', 98, 'UINT32')
    top.set(grp_prefix+'fixed.rate', 6, 'fixedRateEnum') # default 1Hz

    top.set(grp_prefix+'ac.rate', 0, 'acRateEnum')
    for tsnum in range(6):
        top.set(grp_prefix+'ac.ts'+str(tsnum), 0, 'boolEnum')

    top.set(grp_prefix+'eventcode', 0, 'UINT8')

    top.set(grp_prefix+'seq.mode', 15, 'seqEnum')
    top.set(grp_prefix+'seq.burst.mode', 0, 'seqBurstEnum')
    top.set(grp_prefix+'seq.fixed.rate', 0, 'seqFixedRateEnum')
    top.set(grp_prefix+'seq.local.rate', 0, 'seqLocalEnum')

    top.set(grp_prefix+'destination.select', 1, 'destSelectEnum')
    for destnum in range(16):
        top.set(grp_prefix+'destination.dest'+str(destnum), 0, 'boolEnum')

    for inhnum in range(4):
        top.set(grp_prefix+'inhibit'+str(inhnum)+'.enable', 0, 'boolEnum')
        top.set(grp_prefix+'inhibit'+str(inhnum)+'.interval',1,'UINT32')
        top.set(grp_prefix+'inhibit'+str(inhnum)+'.limit',1,'UINT32')

mycdb.modify_device('BEAM', top)
mycdb.print_configs()
