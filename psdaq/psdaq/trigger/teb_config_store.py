from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
import os
import io

# these are the current default values, but put them here to be explicit
create = False
dbname = 'configDB'

args = cdb.createArgs().args
db   = 'configdb' if args.prod else 'devconfigdb'
url  = f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/'

#mycdb = cdb.configdb('https://pswww.slac.stanford.edu/ws-auth/devconfigdb/ws/', args.inst, create,
#                     root=dbname, user=args.user, password=args.password)
mycdb = cdb.configdb(url, args.inst, create,
                     root=dbname, user=args.user, password=args.password)
mycdb.add_alias(args.alias)

# this needs to be called once per detType at the
# "beginning of time" to create the collection name (same as detType
# in top.setInfo).  It doesn't hurt to call it again if the collection
# already exists, however.
mycdb.add_device_config('teb')

top = cdict()

top.setInfo('teb', args.name, args.segm, args.id, 'No comment')
top.setAlg('tebConfig', [0,0,1])

top.set('soname', 'libtmoTeb.so', 'CHARSTR')

top.set('buildAll', 1, 'UINT32')

# This is a required entry:
top.set('prescale', 1, 'UINT32')

top.set('persistValue', 0xdeadbeef, 'UINT32')
top.set('monitorValue', 0x12345678, 'UINT32')

mycdb.add_alias(args.alias)
mycdb.modify_device(args.alias, top)
mycdb.print_configs()
