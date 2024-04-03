from psdaq.configdb.typed_json import cdict
from psdaq.configdb.tsdef import *
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

top['help:RO'] += "\nkeepRawRate: raw data retention rate (Hz)" 

for k in top['user']['SC'].keys():
    top[':types:']['user']['SC'][k]['keepRawRate'] = 'FLOAT'
    top['user']['SC'][k]['keepRawRate'] = 1.

for k,v in top.items():
    print(f'-- key {k}')
    print(top[k])

if not args.dryrun:
    mycdb.modify_device(args.alias, top)

#mycdb.print_configs()
