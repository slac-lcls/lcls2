from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb

# these are the current default values, but I put them here to be explicit
create = True
dbname = 'configDB'

args = cdb.createArgs().args
db   = 'configdb' if args.prod else 'devconfigdb'
url  = f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/'

mycdb = cdb.configdb(url, args.inst, create,
                     root=dbname, user=args.user, password=args.password)

top = mycdb.get_configuration(args.alias, args.name+'_%d'%args.segm)

help_str = "-- user fields --"
help_str += "\nuser.raw.start_ns : nanoseconds from fiducial to sampling start"
help_str += "\nuser.raw.gate_ns  : nanoseconds from sampling start to end"
help_str += "\nuser.raw.prescale : event downsampling; record 1-out-of-N"
help_str += "\nuser.raw.keep     : event downsampling; record on keepraw event"
help_str += "\nuser.fex.start_ns : nanoseconds from fiducial to sparsify start"
help_str += "\nuser.fex.gate_ns  : nanoseconds from sparsify start to end"
help_str += "\nuser.fex.prescale : event downsampling; record 1-out-of-N"
help_str += "\nuser.fex.ymin     : minimum ADC value to sparsify"
help_str += "\nuser.fex.ymax     : maximum ADC value to sparsify"
help_str += "\nuser.fex.xpre     : keep N samples leading excursion"
help_str += "\nuser.fex.xpost    : keep N samples trailing excursion"

top['help:RO'] = help_str

top[':types:']['user']['raw']['keep'] = 'UINT32'
top['user']['raw']['keep'] = 1

top[':types:']['expert']['raw_keep'] = 'UINT32'
top['expert']['raw_keep'] = 0

for k,v in top.items():
    print(f'-- key {k}')
    print(top[k])

mycdb.modify_device(args.alias, top)
#mycdb.print_configs()
