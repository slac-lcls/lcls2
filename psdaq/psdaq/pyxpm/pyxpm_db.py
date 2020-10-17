from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
import os
import io

create = True
dbname = 'configDB'
args = cdb.createArgs().args
db   = 'configdb' if args.prod else 'devconfigdb'
url  = f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/'

mycdb = cdb.configdb(url, args.inst, create,
                     root=dbname, user=args.user, password=args.password)
mycdb.add_device_config('xpm')

top = cdict()

top.setInfo('xpm', args.name, None, 'serial1234', 'No comment')
top.setAlg('config', [0,0,0])

top.set('XTPG.CuDelay'   , 134682, 'UINT32')
top.set('XTPG.CuBeamCode',     40, 'UINT8')
top.set('XTPG.CuInput'   ,      0, 'UINT8')
v = [10]*8
top.set('PART.L0Delay', v, 'UINT32')

if not args.alias in mycdb.get_aliases():
    mycdb.add_alias(args.alias)
mycdb.modify_device(args.alias, top)

mycdb.print_configs()

